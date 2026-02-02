# TRACE
from DDM import DDIM
from BatchManagers import ThreadedScheduler

# Utils and Configs
from Utils import (
    MaskedMSE,
    MovingAverage,
    loadCheckpointFull,
    loadModel,
    restore_training_extras_from_full_checkpoint,
    saveCheckpointFull,
    saveModel,
)
from Configs import *

from device_utils import get_default_device


# torch imports
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# Other imports
from math import isnan
import random
from tqdm import tqdm
import os
import json
from contextlib import nullcontext
import time


def _l2_norm_of_tensors(tensors) -> float:
    total = 0.0
    for t in tensors:
        if t is None:
            continue
        total += float(t.detach().float().norm(2).item()) ** 2
    return total ** 0.5


def _grad_l2_norm(module) -> float | None:
    if module is None:
        return None
    grads = []
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad)
    if not grads:
        return None
    return _l2_norm_of_tensors(grads)


def _param_l2_norm(module) -> float | None:
    if module is None:
        return None
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        return None
    return _l2_norm_of_tensors(params)


def train():
    device = get_default_device()

    # --- Load data ---
    if dataset_name == "apartments":
        dataset = ApartmentsDataset(**dataset_args)
    else:
        dataset = TaxiDataset(**dataset_args)
    steps_per_epoch = max(1, len(dataset) - 2 * batch_size)

    # --- Model and Diffusion Configs ---
    unet = Trace(**Trace_args).to(device).train()
    linkage_shapes = unet.getFeatureShapes(TRAJ_LEN) if hasattr(unet, "getFeatureShapes") else unet.getStateShapes(TRAJ_LEN)
    linkage = Linkage(linkage_shapes, **link_args).to(device).train()
    embedder = Embedder(embed_dim).to(device).train() if embed_dim > 0 else None

    resume_full_ckpt = None
    if resume_checkpoint:
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        try:
            resume_full_ckpt = loadCheckpointFull(resume_checkpoint, map_location="cpu")
        except Exception:
            resume_full_ckpt = None
        resume_models = {"unet": unet, "linkage": linkage}
        if embedder is not None:
            resume_models["embedder"] = embedder
        loadModel(resume_checkpoint, **resume_models)
    diff_manager = DDIM(**diffusion_args, device=device)

    def set_requires_grad(module, requires_grad: bool) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = requires_grad

    # --- Optional: freeze modules for finetune ---
    if freeze_unet:
        set_requires_grad(unet, False)
    if freeze_linkage:
        set_requires_grad(linkage, False)
    if freeze_embedder:
        set_requires_grad(embedder, False)

    # --- Loss function and optimizer ---
    loss_func = MaskedMSE()

    def trainable_params(module):
        if module is None:
            return []
        return [p for p in module.parameters() if p.requires_grad]

    # Embedder should have a smaller learning rate because it is trained on every sample many times.
    params = []
    unet_params = trainable_params(unet)
    if unet_params:
        params.append({"params": unet_params})  # Default learning rate
    linkage_params = trainable_params(linkage)
    if linkage_params:
        params.append({"params": linkage_params})  # Default learning rate
    embedder_params = trainable_params(embedder)
    if embedder_params:
        params.append({"params": embedder_params, "lr": init_lr * 0.01})  # Smaller learning rate

    if not params:
        raise ValueError("No trainable parameters: all modules are frozen.")

    optimizer = optim.AdamW(params, lr=init_lr)

    # --- Recording and Loading ---
    mov_avg_loss = MovingAverage(mov_avg_interval)
    os.makedirs(save_dir)
    writer = SummaryWriter(log_dir)
    info_path = os.path.join(log_dir, "info.txt")
    config_path = os.path.join(log_dir, "train_config.json")
    tb_hist_interval = int(os.environ.get("TRACE_TB_HIST_INTERVAL", "500"))
    writer.add_text("run/log_dir", log_dir, 0)
    writer.add_text("run/save_dir", save_dir, 0)

    run_config = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "T": int(T),
        "TRAJ_LEN": int(TRAJ_LEN),
        "init_lr": float(init_lr),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "diffusion_args": diffusion_args,
        "Trace_args": Trace_args,
        "link_args": link_args,
        "resume_checkpoint": resume_checkpoint,
        "freeze_unet": bool(freeze_unet),
        "freeze_linkage": bool(freeze_linkage),
        "freeze_embedder": bool(freeze_embedder),
        "device": str(device),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    with open(info_path, "w") as file:
        file.write(f"Training {unet.__class__.__name__}\n")
        file.write(f"dataset_name={dataset_name} model_name={model_name}\n")
        file.write(f"init_lr={init_lr} batch_size={batch_size} epochs={epochs}\n")
        file.write(f"diffusion_args={diffusion_args}\n")
        file.write(f"run_config={config_path}\n")
        file.write("Model:\n")
        file.write(str(unet))

    def append_info(line: str) -> None:
        with open(info_path, "a") as file:
            file.write(line + "\n")

    # Restore optimizer/scheduler/RNG and training progress if resuming from a full checkpoint.
    start_global_it = 0
    best_recovery_loss = float("inf")

    batch_manager_class = get_batch_manager_class()
    batch_manager = batch_manager_class(
        ddm=diff_manager,
        skip_step=diffusion_args["skip_step"],
        device=device,
        num_epochs=epochs,
        batch_size=batch_size,
        traj_len=TRAJ_LEN,
        dataset=dataset
    )

    if resume_full_ckpt is not None and hasattr(batch_manager.dataset, "load_state_dict"):
        ds_state = resume_full_ckpt.get("dataset_state")
        if isinstance(ds_state, dict):
            try:
                batch_manager.dataset.load_state_dict(ds_state)
            except Exception as e:
                print(f"warn: failed to restore dataset state ({e}); continue with random sampling.")
                batch_manager.dataset.resetSampleLength(random.choice(list(range(64, 513))))
                batch_manager.dataset.resetEraseRate(random.uniform(0.2, 0.9))
        else:
            batch_manager.dataset.resetSampleLength(random.choice(list(range(64, 513))))
            batch_manager.dataset.resetEraseRate(random.uniform(0.2, 0.9))
    else:
        batch_manager.dataset.resetSampleLength(random.choice(list(range(64, 513))))
        batch_manager.dataset.resetEraseRate(random.uniform(0.2, 0.9))

    # Register states
    for shape in unet.getStateShapes(TRAJ_LEN):
        batch_manager.registerState(shape)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                     factor=lr_reduce_factor,
                                     patience=int(lr_reduce_patience),
                                     min_lr=1e-6)

    if resume_full_ckpt is not None:
        train_state = restore_training_extras_from_full_checkpoint(
            resume_full_ckpt,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device if isinstance(device, torch.device) else torch.device(device),
            restore_rng_state=True,
        )
        if isinstance(train_state.get("best_recovery_loss"), (int, float)):
            best_recovery_loss = float(train_state["best_recovery_loss"])
        ma_state = train_state.get("mov_avg_loss")
        if isinstance(ma_state, dict):
            mov_avg_loss.load_state_dict(ma_state)
        if isinstance(train_state.get("global_it"), int):
            start_global_it = int(train_state["global_it"]) + 1

        bm_state = resume_full_ckpt.get("batch_manager_state")
        if isinstance(bm_state, dict) and hasattr(batch_manager, "load_state_dict"):
            try:
                batch_manager.load_state_dict(bm_state)
            except Exception as e:
                print(f"warn: failed to restore batch_manager state ({e}); continue with new iterator state.")

    # --- Training Loop ---

    # The proposed training algorithm with 2 denoising steps trained in one iteration
    recovery_eval_interval = 1000
    use_prefetch = os.environ.get("TRACE_PREFETCH", "1").strip().lower() not in ("0", "false", "no")
    scheduler_cm = ThreadedScheduler(batch_manager, 3) if use_prefetch else nullcontext(batch_manager)

    with scheduler_cm as data_iterator:
        total_iterations_all = epochs * steps_per_epoch
        if start_global_it >= total_iterations_all:
            append_info(f"Resume checkpoint already at/over target iters: start_global_it={start_global_it} total={total_iterations_all}")
            writer.flush()
            writer.close()
            return

        remaining_iterations = total_iterations_all - start_global_it
        # Make iterator length match the remaining iterations so tqdm and ThreadedScheduler won't hang.
        if hasattr(batch_manager, "total_iterations"):
            batch_manager.total_iterations = remaining_iterations

        total_iterations = len(data_iterator)
        pbar = tqdm(data_iterator, desc="Training", ncols=100, total=total_iterations)
        last_tb_wall = time.perf_counter()
        last_tb_it = start_global_it
        global_it = start_global_it
        for batch_data in pbar:
            iter_t0 = time.perf_counter()
            optimizer.zero_grad()
            epoch = global_it // steps_per_epoch + 1
            iter_in_epoch = global_it % steps_per_epoch + 1

            if dataset_name == "apartments":
                t, tp1, x_t, x_tp1, x_T, eps_0_to_t, eps_0_to_tp1, masks, loc_mean, meta, s_tp1 = batch_data
                embed = embedder(meta, loc_mean)
                output_tp1, hidden = unet(torch.cat([x_tp1, embed], dim=1), tp1, s_tp1)
                s_t = linkage(hidden, s_tp1, tp1)
                output_t, _ = unet(torch.cat([x_t, embed], dim=1), t, s_t)
            else:
                t, tp1, x_t, x_tp1, x_T, eps_0_to_t, eps_0_to_tp1, masks, s_tp1 = batch_data
                output_tp1, hidden = unet(x_tp1, tp1, s_tp1)
                s_t = linkage(hidden, s_tp1, tp1)
                output_t, _ = unet(x_t, t, s_t)

            # Loss for the first denoising step & embedding
            loss_tp1 = loss_func(output_tp1, eps_0_to_tp1, masks)
            # Loss for the second denoising step & embedding & linkage
            loss_t = loss_func(output_t, eps_0_to_t, masks)
            loss = loss_tp1 + loss_t

            loss_float = loss.item()
            loss_tp1_float = loss_tp1.item()
            loss_t_float = loss_t.item()

            if isnan(loss_float):
                pbar.set_postfix_str("Nan Loss detected")
                global_it += 1
                continue

            # At the beginning, all states are initialize to 0 regardless of the actual diffusion step
            # However, state should only be 0 at t=T-1
            # So we don't back-propagate the loss until the states values are updated
            if global_it > max(actual_diff_step, batch_size):
                loss.backward()
                optimizer.step()

            batch_manager.updateState(s_t)

            mov_avg_loss << loss_float

            if global_it % 10 == 0:
                pbar.set_postfix_str(
                    f"epoch={epoch}/{epochs} iter={iter_in_epoch}/{steps_per_epoch} "
                    f"loss={float(mov_avg_loss):.7f} | lr={optimizer.param_groups[0]['lr']:.4e}"
                )

            if global_it % log_interval == 0:
                # Basic scalars
                # Backward compatibility: keep the old tags.
                writer.add_scalar("Loss", float(mov_avg_loss), global_it)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_it)
                writer.add_scalar("Loss/mov_avg", float(mov_avg_loss), global_it)
                writer.add_scalar("Loss/raw", float(loss_float), global_it)
                writer.add_scalar("Loss/tp1", float(loss_tp1_float), global_it)
                writer.add_scalar("Loss/t", float(loss_t_float), global_it)

                # Per-param-group LR
                for gi, group in enumerate(optimizer.param_groups):
                    if "lr" in group:
                        writer.add_scalar(f"LR/group{gi}", float(group["lr"]), global_it)

                # Diffusion step stats
                writer.add_scalar("Diffusion/t_mean", float(t.float().mean().item()), global_it)
                writer.add_scalar("Diffusion/t_min", float(t.min().item()), global_it)
                writer.add_scalar("Diffusion/t_max", float(t.max().item()), global_it)
                writer.add_scalar("Diffusion/tp1_mean", float(tp1.float().mean().item()), global_it)
                writer.add_scalar("Diffusion/tp1_min", float(tp1.min().item()), global_it)
                writer.add_scalar("Diffusion/tp1_max", float(tp1.max().item()), global_it)

                # Mask/data stats (from the erase_mask channel inside model input)
                # x_tp1: (B, 6, L) for both taxi and apartments, last channel is erase_mask (1=erase, 0=observe, -1=padding)
                erase_mask = x_tp1[:, 5, :]
                valid_1d = erase_mask >= 0
                erased_1d = (erase_mask > 0.1) & valid_1d

                valid_counts = valid_1d.sum(dim=1).float()  # (B,)
                erased_counts = erased_1d.sum(dim=1).float()  # (B,)
                erase_rate_per = torch.where(valid_counts > 0, erased_counts / valid_counts, torch.zeros_like(valid_counts))

                writer.add_scalar("Data/points_valid_mean", float(valid_counts.mean().item()), global_it)
                writer.add_scalar("Data/points_valid_median", float(valid_counts.median().item()), global_it)
                writer.add_scalar("Data/points_erased_mean", float(erased_counts.mean().item()), global_it)
                writer.add_scalar("Data/erase_rate_mean", float(erase_rate_per.mean().item()), global_it)
                writer.add_scalar("Data/erase_rate_median", float(erase_rate_per.median().item()), global_it)
                if tb_hist_interval > 0 and global_it % tb_hist_interval == 0:
                    writer.add_histogram("Data/sample_length", valid_counts, global_it)
                    writer.add_histogram("Data/erase_rate", erase_rate_per, global_it)

                # Output/target magnitude (helps spot scale explosions)
                writer.add_scalar("Eps/output_tp1_abs_mean", float(output_tp1.detach().abs().mean().item()), global_it)
                writer.add_scalar("Eps/target_tp1_abs_mean", float(torch.stack(eps_0_to_tp1).detach().abs().mean().item()), global_it)
                writer.add_scalar("Eps/output_t_abs_mean", float(output_t.detach().abs().mean().item()), global_it)
                writer.add_scalar("Eps/target_t_abs_mean", float(torch.stack(eps_0_to_t).detach().abs().mean().item()), global_it)

                # Grad/param norms (trainable only)
                g_unet = _grad_l2_norm(unet)
                g_link = _grad_l2_norm(linkage)
                g_emb = _grad_l2_norm(embedder)
                if g_unet is not None:
                    writer.add_scalar("GradNorm/unet", float(g_unet), global_it)
                if g_link is not None:
                    writer.add_scalar("GradNorm/linkage", float(g_link), global_it)
                if g_emb is not None:
                    writer.add_scalar("GradNorm/embedder", float(g_emb), global_it)
                if g_unet is not None or g_link is not None or g_emb is not None:
                    g_total = ((g_unet or 0.0) ** 2 + (g_link or 0.0) ** 2 + (g_emb or 0.0) ** 2) ** 0.5
                    writer.add_scalar("GradNorm/total", float(g_total), global_it)

                if progress_log_interval > 0 and global_it % progress_log_interval == 0:
                    p_unet = _param_l2_norm(unet)
                    p_link = _param_l2_norm(linkage)
                    p_emb = _param_l2_norm(embedder)
                    if p_unet is not None:
                        writer.add_scalar("ParamNorm/unet", float(p_unet), global_it)
                    if p_link is not None:
                        writer.add_scalar("ParamNorm/linkage", float(p_link), global_it)
                    if p_emb is not None:
                        writer.add_scalar("ParamNorm/embedder", float(p_emb), global_it)

                # Timing
                wall_now = time.perf_counter()
                dt = wall_now - last_tb_wall
                dit = max(1, global_it - last_tb_it)
                if dt > 0:
                    writer.add_scalar("Time/iter_per_sec", float(dit / dt), global_it)
                    writer.add_scalar("Time/sec_per_iter", float(dt / dit), global_it)
                writer.add_scalar("Time/iter_wall_sec", float(time.perf_counter() - iter_t0), global_it)
                last_tb_wall = wall_now
                last_tb_it = global_it

                # CUDA memory stats (if applicable)
                if isinstance(device, torch.device) and device.type == "cuda":
                    writer.add_scalar("CUDA/max_memory_allocated_mb", float(torch.cuda.max_memory_allocated(device) / 1024**2), global_it)
                    writer.add_scalar("CUDA/memory_reserved_mb", float(torch.cuda.memory_reserved(device) / 1024**2), global_it)

            if progress_log_interval > 0 and global_it % progress_log_interval == 0:
                append_info(
                    f"iter={global_it} epoch={epoch}/{epochs} "
                    f"iter_in_epoch={iter_in_epoch}/{steps_per_epoch} "
                    f"loss={float(mov_avg_loss):.7f} lr={optimizer.param_groups[0]['lr']:.4e}"
                )

            # Skip the very first iteration to avoid long I/O pauses at startup.
            if global_it > 0 and global_it % 500 == 0:
                lr_scheduler.step(float(mov_avg_loss))
                saveModel(save_dir + "last.pth", async_write=True, unet=unet, linkage=linkage, embedder=embedder)
                full_models = {"unet": unet, "linkage": linkage, "embedder": embedder}
                saveCheckpointFull(
                    save_dir + "last_full.pth",
                    async_write=True,
                    models=full_models,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    train_state={
                        "global_it": int(global_it),
                        "epoch": int(epoch),
                        "iter_in_epoch": int(iter_in_epoch),
                        "steps_per_epoch": int(steps_per_epoch),
                        "epochs": int(epochs),
                        "best_recovery_loss": float(best_recovery_loss),
                        "mov_avg_loss": mov_avg_loss.state_dict(),
                    },
                    dataset_state=batch_manager.dataset.state_dict() if hasattr(batch_manager.dataset, "state_dict") else None,
                    batch_manager_state=batch_manager.state_dict() if hasattr(batch_manager, "state_dict") else None,
                    run_config=run_config,
                )

            if save_ckpt_interval > 0 and global_it % save_ckpt_interval == 0 and global_it > 0:
                saveModel(
                    save_dir + f"ckpt_e{epoch}_i{iter_in_epoch}_g{global_it}.pth",
                    async_write=True,
                    unet=unet,
                    linkage=linkage,
                    embedder=embedder
                )

            # Skip the very first iteration to avoid heavy recovery eval right at startup.
            # For short runs (e.g. small datasets / low epochs), also evaluate at the final iteration.
            is_last_iteration = global_it == total_iterations_all - 1
            if global_it > 0 and (global_it % recovery_eval_interval == 0 or is_last_iteration):
                try:
                    from eval import recovery
                except ModuleNotFoundError as e:
                    print(f"Skip recovery eval: missing dependency ({e})")
                    recovery = None

                if recovery is not None:
                    try:
                        recovery_loss, fig = recovery(diff_manager, unet, linkage, embedder)
                    except FileNotFoundError as e:
                        print(f"Skip recovery eval: missing test file ({e})")
                        recovery_loss, fig = None, None

                    if recovery_loss is not None:
                        writer.add_scalar("Recovery Loss", recovery_loss, global_it)
                        writer.add_figure("Recovery Figure", fig, global_it)

                        if recovery_loss < best_recovery_loss:
                            best_recovery_loss = recovery_loss
                            saveModel(save_dir + "best.pth", async_write=True, unet=unet, linkage=linkage, embedder=embedder)
                            full_models = {"unet": unet, "linkage": linkage, "embedder": embedder}
                            saveCheckpointFull(
                                save_dir + "best_full.pth",
                                async_write=True,
                                models=full_models,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                train_state={
                                    "global_it": int(global_it),
                                    "epoch": int(epoch),
                                    "iter_in_epoch": int(iter_in_epoch),
                                    "steps_per_epoch": int(steps_per_epoch),
                                    "epochs": int(epochs),
                                    "best_recovery_loss": float(best_recovery_loss),
                                    "mov_avg_loss": mov_avg_loss.state_dict(),
                                },
                                dataset_state=batch_manager.dataset.state_dict() if hasattr(batch_manager.dataset, "state_dict") else None,
                                batch_manager_state=batch_manager.state_dict() if hasattr(batch_manager, "state_dict") else None,
                                run_config=run_config,
                            )

            global_it += 1

    writer.flush()
    writer.close()

if __name__ == "__main__":
    train()
