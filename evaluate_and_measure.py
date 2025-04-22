import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
# from net.CIDNet import CIDNet
from net.CIDNet_MSSA import CIDNet
import torch
import glob
import cv2
import lpips
import numpy as np
from PIL import Image
import platform

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Evaluate and Measure')
# Eval arguments
parser.add_argument('--perc', action='store_true', help='Use weights trained with perceptual loss')
parser.add_argument('--lol', action='store_true', help='Evaluate/Measure LOLv1 dataset')
parser.add_argument('--lol_v2_real', action='store_true', help='Evaluate/Measure LOLv2 real dataset')
parser.add_argument('--lol_v2_syn', action='store_true', help='Evaluate/Measure LOLv2 synthetic dataset')
parser.add_argument('--SICE_grad', action='store_true', help='Evaluate/Measure SICE_grad dataset')
parser.add_argument('--SICE_mix', action='store_true', help='Evaluate/Measure SICE_mix dataset')
parser.add_argument('--lmot', action='store_true', help='Evaluate/Measure LMOT dataset')

parser.add_argument('--best_GT_mean', action='store_true', help='Use LOLv2 real dataset best_GT_mean weights and settings')
parser.add_argument('--best_PSNR', action='store_true', help='Use LOLv2 real dataset best_PSNR weights and settings')
parser.add_argument('--best_SSIM', action='store_true', help='Use LOLv2 real dataset best_SSIM weights and settings')

parser.add_argument('--custome', action='store_true', help='Evaluate/Measure custom dataset')
parser.add_argument('--custome_path', type=str, default='./YOLO', help='Path to custom low-light images for unpaired evaluation')
parser.add_argument('--unpaired', action='store_true', help='Evaluate unpaired datasets (DICM, LIME, MEF, NPE, VV, custom)')
parser.add_argument('--DICM', action='store_true', help='Evaluate/Measure DICM dataset (unpaired)')
parser.add_argument('--LIME', action='store_true', help='Evaluate/Measure LIME dataset (unpaired)')
parser.add_argument('--MEF', action='store_true', help='Evaluate/Measure MEF dataset (unpaired)')
parser.add_argument('--NPE', action='store_true', help='Evaluate/Measure NPE dataset (unpaired)')
parser.add_argument('--VV', action='store_true', help='Evaluate/Measure VV dataset (unpaired)')
parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value for LOLv2/unpaired evaluation')
parser.add_argument('--gamma', type=float, default=1.0, help='Gamma correction value for input')
parser.add_argument('--unpaired_weights', type=str, default='./weights/LOLv2_syn/w_perc.pth', help='Weights for unpaired evaluation')

# Measure arguments
parser.add_argument('--use_GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model during measurement')

args = parser.parse_args()


# --- Evaluation Function (from eval.py) ---
def eval_model(model, testing_data_loader, model_path, output_folder, norm_size=True, LOL=False, v2=False, unpaired=False, alpha=1.0, gamma=1.0, lmot=False):
    print(f"Loading weights from: {model_path}")
    torch.set_grad_enabled(False)
    try:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        print('Pre-trained model loaded successfully.')
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {e}")
        raise
    model.eval()
    print('Starting Evaluation...')
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    elif lmot:
        model.trans.gated = True # Use LOLv1 setting for LMOT, adjust if needed

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output directory: {output_folder}")
    else:
         print(f"Output directory already exists: {output_folder}")


    for batch in tqdm(testing_data_loader, desc=f"Evaluating"):
        with torch.no_grad():
            if norm_size:
                input_tensor, name = batch[0], batch[1]
            else:
                input_tensor, name, h, w = batch[0], batch[1], batch[2], batch[3]

            input_tensor = input_tensor.cuda()
            output = model(input_tensor**gamma)

        output = torch.clamp(output.cuda(), 0, 1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]

        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(os.path.join(output_folder, name[0]))
        torch.cuda.empty_cache()

    print('===> Evaluation Finished')
    # Reset model gates if they were modified
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    elif lmot:
        model.trans.gated = False
    torch.set_grad_enabled(True)


# --- Measurement Functions (from measure.py) ---
def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        # Try resizing prediction to match reference
        print(f"Warning: SSIM shape mismatch {img1.shape} vs {img2.shape}. Resizing prediction.")
        target_pil = Image.fromarray(target.astype(np.uint8))
        ref_pil = Image.fromarray(ref.astype(np.uint8))
        target_pil = target_pil.resize(ref_pil.size)
        img1 = np.array(target_pil, dtype=np.float64)
        # raise ValueError('Input images must have the same dimensions.')

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    if not img1.shape == img2.shape:
         # Try resizing prediction to match reference
        print(f"Warning: PSNR shape mismatch {img1.shape} vs {img2.shape}. Resizing prediction.")
        target_pil = Image.fromarray(target.astype(np.uint8))
        ref_pil = Image.fromarray(ref.astype(np.uint8))
        target_pil = target_pil.resize(ref_pil.size)
        img1 = np.array(target_pil, dtype=np.float32)
        # raise ValueError('Input images must have the same dimensions.')

    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8))
    return psnr

def metrics(im_dir_pattern, label_dir, use_GT_mean_flag):
    print(f"Starting Measurement on: {im_dir_pattern}")
    print(f"Using Label directory: {label_dir}")
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    n = 0
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()

    if not label_dir.endswith(os.path.sep):
        label_dir += os.path.sep

    is_lmot = 'lmot_lol_val/img_light_rgb' in label_dir # Heuristic check for LMOT

    image_files = sorted(glob.glob(im_dir_pattern))
    if not image_files:
        print(f"Error: No images found matching pattern {im_dir_pattern}")
        return 0, 0, 0

    for item in tqdm(image_files, desc="Measuring"):
        try:
            im1_pil = Image.open(item).convert('RGB') # Prediction
        except Exception as e:
             print(f"Error opening prediction image {item}: {e}")
             continue

        # Determine name based on OS
        os_name = platform.system()
        if os_name.lower() == 'windows':
            name = item.split('\')[-1]
        else: # Linux or other POSIX
            name = item.split('/')[-1]

        name_without_ext = os.path.splitext(name)[0]

        # Find corresponding ground truth image
        ref_path = None
        possible_exts = ['.png', '.jpg', '.JPG', '.jpeg', '.JPEG', '.PNG'] # Add more if needed
        if is_lmot:
             # LMOT specifically uses .jpg
             potential_path = os.path.join(label_dir, name_without_ext + '.jpg')
             if os.path.exists(potential_path):
                 ref_path = potential_path
        else:
            # Try finding GT with the same name or common extensions
            if os.path.exists(os.path.join(label_dir, name)):
                 ref_path = os.path.join(label_dir, name)
            else:
                for ext in possible_exts:
                    potential_path = os.path.join(label_dir, name_without_ext + ext)
                    if os.path.exists(potential_path):
                        ref_path = potential_path
                        break

        if ref_path is None:
            print(f"Warning: Could not find reference image for {name} in {label_dir}")
            continue

        try:
            im2_pil = Image.open(ref_path).convert('RGB') # Ground Truth
        except Exception as e:
            print(f"Error opening reference image {ref_path}: {e}")
            continue

        # Ensure dimensions match - Resize prediction to GT size
        if im1_pil.size != im2_pil.size:
            print(f"Warning: Resizing prediction {name} from {im1_pil.size} to GT size {im2_pil.size}")
            im1_pil = im1_pil.resize(im2_pil.size)

        im1 = np.array(im1_pil) # Prediction array
        im2 = np.array(im2_pil) # Ground Truth array

        # Optional GT Mean rectification
        if use_GT_mean_flag:
            try:
                mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
                mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
                if mean_restored > 1e-5: # Avoid division by zero or near-zero
                    im1 = np.clip(im1 * (mean_target / mean_restored), 0, 255)
                else:
                    print(f"Warning: Near-zero mean for prediction {name}, skipping GT mean rectification.")
            except Exception as e:
                 print(f"Error during GT mean rectification for {name}: {e}")


        # Calculate metrics
        try:
            score_psnr = calculate_psnr(im1, im2)
            score_ssim = calculate_ssim(im1, im2)
            # LPIPS calculation
            with torch.no_grad():
                ex_p0 = lpips.im2tensor(im1).cuda() # Prediction tensor
                ex_ref = lpips.im2tensor(im2).cuda() # Reference tensor
                score_lpips_tensor = loss_fn.forward(ex_ref, ex_p0)
            score_lpips = score_lpips_tensor.item()

            avg_psnr += score_psnr
            avg_ssim += score_ssim
            avg_lpips += score_lpips
            n += 1
            torch.cuda.empty_cache()
        except Exception as e:
             print(f"Error calculating metrics for {name}: {e}")
             continue


    if n == 0:
        print(f"Error: Failed to evaluate any images successfully from {im_dir_pattern}")
        return 0, 0, 0

    avg_psnr /= n
    avg_ssim /= n
    avg_lpips /= n
    print("===> Measurement Finished")
    return avg_psnr, avg_ssim, avg_lpips

# --- Main Execution Block ---
if __name__ == '__main__':

    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")

    if not os.path.exists('./output'):
        os.makedirs('./output', exist_ok=True)

    # --- Configuration based on arguments ---
    eval_data_path = None
    label_dir = None # Ground truth directory for measurement
    output_folder = None
    weight_path = None
    norm_size = True
    num_workers = 4 # Can be adjusted
    alpha_val = args.alpha # Use args.alpha by default
    gamma_val = args.gamma
    run_eval = True # Flag to control evaluation step
    run_measure = True # Flag to control measurement step

    # Dataset specific flags for eval function
    is_lol, is_v2, is_unpaired, is_lmot_flag = False, False, False, False

    if args.lol:
        is_lol = True
        eval_data_path = "./datasets/LOLdataset/eval15/low"
        label_dir = './datasets/LOLdataset/eval15/high/'
        output_folder = './output/LOLv1/'
        weight_path = './weights/LOLv1/w_perc.pth' if args.perc else './weights/LOLv1/wo_perc.pth'

    elif args.lol_v2_real:
        is_v2 = True
        eval_data_path = "./datasets/LOLv2/Real_captured/Test/Low"
        label_dir = './datasets/LOLv2/Real_captured/Test/Normal/'
        output_folder = './output/LOLv2_real/'
        if args.best_GT_mean:
            weight_path = './weights/LOLv2_real/w_perc.pth' # Assuming this is the intended weight
            alpha_val = 0.84
        elif args.best_PSNR:
            weight_path = './weights/LOLv2_real/best_PSNR.pth'
            alpha_val = 0.8
        elif args.best_SSIM:
            weight_path = './weights/LOLv2_real/best_SSIM.pth'
            alpha_val = 0.82
        else: # Default if no specific best_* flag is set
            weight_path = './weights/LOLv2_real/w_perc.pth' if args.perc else './weights/LOLv2_real/wo_perc.pth'
            alpha_val = args.alpha # Use default or specified alpha

    elif args.lol_v2_syn:
        is_v2 = True
        eval_data_path = "./datasets/LOLv2/Synthetic/Test/Low"
        label_dir = './datasets/LOLv2/Synthetic/Test/Normal/'
        output_folder = './output/LOLv2_syn/'
        weight_path = './weights/LOLv2_syn/w_perc.pth' if args.perc else './weights/LOLv2_syn/wo_perc.pth'

    elif args.SICE_grad:
        eval_data_path = "./datasets/SICE/SICE_Grad"
        label_dir = './datasets/SICE/SICE_Reshape/' # Assuming GT is in Reshape
        output_folder = './output/SICE_grad/'
        weight_path = './weights/SICE.pth' # Common weight for SICE?
        norm_size = False
        # Note: Measurement might require specific handling if GT is not paired per image.

    elif args.SICE_mix:
        eval_data_path = "./datasets/SICE/SICE_Mix"
        label_dir = './datasets/SICE/SICE_Reshape/' # Assuming GT is in Reshape
        output_folder = './output/SICE_mix/'
        weight_path = './weights/SICE.pth' # Common weight for SICE?
        norm_size = False
        # Note: Measurement might require specific handling if GT is not paired per image.

    elif args.lmot:
        is_lmot_flag = True
        eval_data_path = "/root/autodl-tmp/lmot_lol_val/img_dark_rgb" # Make sure this path is correct
        label_dir = '/root/autodl-tmp/lmot_lol_val/img_light_rgb' # Make sure this path is correct
        output_folder = './output/LMOT/'
        # Adjust weight path as needed
        weight_path = '/root/HVI-CIDNet/weights/MSSA_P1en2/epoch_30.pth' if args.perc else './weights/LMOT/wo_perc.pth' # Example

    elif args.unpaired:
        is_unpaired = True
        norm_size = False
        weight_path = args.unpaired_weights
        alpha_val = args.alpha
        run_measure = False # Cannot measure unpaired data without GT
        print("Unpaired dataset selected. Measurement step will be skipped.")

        if args.DICM:
            eval_data_path = "./datasets/DICM"
            output_folder = './output/DICM/'
        elif args.LIME:
            eval_data_path = "./datasets/LIME"
            output_folder = './output/LIME/'
        elif args.MEF:
            eval_data_path = "./datasets/MEF"
            output_folder = './output/MEF/'
        elif args.NPE:
            eval_data_path = "./datasets/NPE"
            output_folder = './output/NPE/'
        elif args.VV:
            eval_data_path = "./datasets/VV"
            output_folder = './output/VV/'
        elif args.custome:
            eval_data_path = args.custome_path
            output_folder = './output/custome/'
        else:
             run_eval = False
             print("Error: --unpaired requires a specific dataset flag (e.g., --DICM, --LIME, --custome)")


    # --- Execute Steps ---
    if run_eval and eval_data_path and weight_path and output_folder:
        print("-" * 20)
        print("--- Starting Evaluation Phase ---")
        print(f"Dataset Path: {eval_data_path}")
        print(f"Weight Path: {weight_path}")
        print(f"Output Folder: {output_folder}")
        print(f"Normalization: {norm_size}")
        print(f"Alpha: {alpha_val}, Gamma: {gamma_val}")
        print(f"Flags: LOL={is_lol}, v2={is_v2}, unpaired={is_unpaired}, LMOT={is_lmot_flag}")
        print("-" * 20)


        # Choose dataset loading function based on dataset type
        if args.SICE_grad or args.SICE_mix or args.unpaired:
             eval_dataset = get_SICE_eval_set(eval_data_path)
        else: # LOL, LOLv2, LMOT use get_eval_set
             eval_dataset = get_eval_set(eval_data_path)

        eval_data_loader = DataLoader(dataset=eval_dataset, num_workers=num_workers, batch_size=1, shuffle=False)

        # Initialize Model
        eval_net = CIDNet().cuda()

        # Run Evaluation
        eval_model(eval_net, eval_data_loader, weight_path, output_folder,
                   norm_size=norm_size, LOL=is_lol, v2=is_v2, unpaired=is_unpaired,
                   alpha=alpha_val, gamma=gamma_val, lmot=is_lmot_flag)

    elif run_eval:
        print("Evaluation skipped - Configuration incomplete (check dataset/weight paths).")


    if run_measure and label_dir and output_folder:
        print("-" * 20)
        print("--- Starting Measurement Phase ---")
        print(f"Evaluating images in: {output_folder}")
        print(f"Comparing against GT in: {label_dir}")
        print(f"Use GT Mean Rectification: {args.use_GT_mean}")
        print("-" * 20)

        im_dir_pattern = os.path.join(output_folder, '*.png') # Assuming output is always png
        avg_psnr, avg_ssim, avg_lpips = metrics(im_dir_pattern, label_dir, args.use_GT_mean)

        print("--- Measurement Results ---")
        print(f"Dataset: {output_folder.split('/')[-2] if output_folder else 'N/A'}") # Extract dataset name from path
        print("===> Average PSNR: {:.4f} dB ".format(avg_psnr))
        print("===> Average SSIM: {:.4f} ".format(avg_ssim))
        print("===> Average LPIPS: {:.4f} ".format(avg_lpips))
        print("-" * 25)

    elif run_measure:
         print("Measurement skipped - Configuration incomplete or dataset is unpaired.")

    print("Script finished.") 