# TRACE
from DDM import DDIM
from BatchManagers import ThreadedScheduler

# Utils and Configs
from Utils import MovingAverage, loadModel, saveModel, MaskedMSE
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
    if resume_checkpoint:
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
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

    batch_manager.dataset.resetSampleLength(random.choice(list(range(64, 513))))
    batch_manager.dataset.resetEraseRate(random.uniform(0.2, 0.9))

    # Register states
    for shape in unet.getStateShapes(TRAJ_LEN):
        batch_manager.registerState(shape)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                     factor=lr_reduce_factor,
                                     patience=int(lr_reduce_patience),
                                     min_lr=1e-6)

    # --- Training Loop ---

    # The proposed training algorithm with 2 denoising steps trained in one iteration
    best_recovery_loss = float("inf")
    recovery_eval_interval = 1000
    use_prefetch = os.environ.get("TRACE_PREFETCH", "1").strip().lower() not in ("0", "false", "no")
    scheduler_cm = ThreadedScheduler(batch_manager, 3) if use_prefetch else nullcontext(batch_manager)

    with scheduler_cm as data_iterator:
        total_iterations = len(data_iterator)
        pbar = tqdm(data_iterator, desc="Training", ncols=100)
        for global_it, batch_data in enumerate(pbar):
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

            if isnan(loss_float):
                pbar.set_postfix_str("Nan Loss detected")
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
                writer.add_scalar("Loss", float(mov_avg_loss), global_it)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_it)

            if progress_log_interval > 0 and global_it % progress_log_interval == 0:
                append_info(
                    f"iter={global_it} epoch={epoch}/{epochs} "
                    f"iter_in_epoch={iter_in_epoch}/{steps_per_epoch} "
                    f"loss={float(mov_avg_loss):.7f} lr={optimizer.param_groups[0]['lr']:.4e}"
                )

            # Skip the very first iteration to avoid long I/O pauses at startup.
            if global_it > 0 and global_it % 500 == 0:
                saveModel(save_dir + "last.pth", async_write=True, unet=unet, linkage=linkage, embedder=embedder)
                lr_scheduler.step(float(mov_avg_loss))

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
            is_last_iteration = global_it == total_iterations - 1
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

                    if recovery_loss is None:
                        continue
                    writer.add_scalar("Recovery Loss", recovery_loss, global_it)
                    writer.add_figure("Recovery Figure", fig, global_it)

                    if recovery_loss < best_recovery_loss:
                        best_recovery_loss = recovery_loss
                        saveModel(save_dir + "best.pth", async_write=True, unet=unet, linkage=linkage, embedder=embedder)

if __name__ == "__main__":
    train()
