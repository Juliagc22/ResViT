import os
import csv
import numpy as np

try:
    from skimage.metrics import (
        peak_signal_noise_ratio,
        structural_similarity,
    )
except ImportError:
    # Compatibilitat amb versions antigues de scikit-image
    from skimage.measure import compare_psnr as peak_signal_noise_ratio
    from skimage.measure import compare_ssim as structural_similarity

from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    metrics_rows = []
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()

        # ---------------------------------------------------------
        # Obtenir predicció i ground truth directament dels tensors
        # ---------------------------------------------------------
        fake_im = model.fake_B.detach().cpu().numpy().astype(np.float32)
        real_im = model.real_B.detach().cpu().numpy().astype(np.float32)

        # En aquest experiment output_nc=1:
        # [batch, canal, altura, amplada] -> [altura, amplada]
        fake_im = np.squeeze(fake_im)
        real_im = np.squeeze(real_im)

        # ResViT treballa habitualment en l'interval [-1, 1].
        # Convertim les imatges a [0, 1] abans de calcular mètriques.
        fake_im = np.clip(fake_im * 0.5 + 0.5, 0.0, 1.0)
        real_im = np.clip(real_im * 0.5 + 0.5, 0.0, 1.0)

        # Evitar mostres completament buides
        if real_im.max() > 0:
            mae_value = float(
                np.mean(np.abs(fake_im - real_im))
            )

            psnr_value = float(
                peak_signal_noise_ratio(
                    real_im,
                    fake_im,
                    data_range=1.0,
                )
            )

            ssim_value = float(
                structural_similarity(
                    real_im,
                    fake_im,
                    data_range=1.0,
                )
            )

            current_path = model.get_image_paths()

            if isinstance(current_path, (list, tuple)):
                image_name = os.path.basename(current_path[0])
            else:
                image_name = os.path.basename(str(current_path))

            metrics_rows.append({
                "index": i,
                "image": image_name,
                "mae": mae_value,
                "psnr": psnr_value,
                "ssim": ssim_value,
            })

            print(
                f"{i:04d} | "
                f"MAE: {mae_value:.6f} | "
                f"PSNR: {psnr_value:.3f} | "
                f"SSIM: {ssim_value:.4f}"
            )
        
        if opt.dataset_mode=='aligned_mat':
            visuals=model.get_current_visuals()
            #visuals['real_A']=visuals['real_A'][:,:,0:3]
            #visuals['real_B']=visuals['real_B'][:,:,0:3]
            #visuals['fake_B']=visuals['fake_B'][:,:,0:3]    
            img_path = model.get_image_paths()
            img_path[0]=img_path[0]+str(i)
        elif  opt.dataset_mode=='unaligned_mat':   
            visuals=model.get_current_visuals()
            slice_select=[opt.input_nc/2,opt.input_nc/2,opt.input_nc/2]
            visuals['real_A']=visuals['real_A'][:,:,slice_select]
            visuals['real_B']=visuals['real_B'][:,:,slice_select]
            visuals['fake_A']=visuals['fake_A'][:,:,slice_select]
            visuals['fake_B']=visuals['fake_B'][:,:,slice_select]
            visuals['rec_A']=visuals['rec_A'][:,:,slice_select]
            visuals['rec_B']=visuals['rec_B'][:,:,slice_select]
            #temp_visuals['idt_A']=temp_visuals['idt_A'][:,:,slice_select]
            #temp_visuals['idt_B']=temp_visuals['idt_B'][:,:,slice_select]                    
            img_path = model.get_image_paths()
            img_path[0]=img_path[0]+str(i)            
        else:
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    webpage.save()

# ---------------------------------------------------------
# Desar mètriques individuals i resum
# ---------------------------------------------------------
if metrics_rows:
    os.makedirs(web_dir, exist_ok=True)

    csv_path = os.path.join(web_dir, "metrics.csv")

    mae_values = np.asarray(
        [row["mae"] for row in metrics_rows],
        dtype=np.float64,
    )
    psnr_values = np.asarray(
        [row["psnr"] for row in metrics_rows],
        dtype=np.float64,
    )
    ssim_values = np.asarray(
        [row["ssim"] for row in metrics_rows],
        dtype=np.float64,
    )

    summary = {
        "index": "MEAN",
        "image": f"{len(metrics_rows)} images",
        "mae": float(np.mean(mae_values)),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }

    std_summary = {
        "index": "STD",
        "image": "",
        "mae": float(np.std(mae_values)),
        "psnr": float(np.std(psnr_values)),
        "ssim": float(np.std(ssim_values)),
    }

    fieldnames = [
        "index",
        "image",
        "mae",
        "psnr",
        "ssim",
    ]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(metrics_rows)
        writer.writerow(summary)
        writer.writerow(std_summary)

    print("\n================ TEST RESULTS ================")
    print(f"Number of images: {len(metrics_rows)}")
    print(
        f"MAE:  {summary['mae']:.6f} "
        f"± {std_summary['mae']:.6f}"
    )
    print(
        f"PSNR: {summary['psnr']:.3f} "
        f"± {std_summary['psnr']:.3f} dB"
    )
    print(
        f"SSIM: {summary['ssim']:.4f} "
        f"± {std_summary['ssim']:.4f}"
    )
    print(f"Metrics saved in: {csv_path}")
else:
    print("No valid images were available for metric calculation.")