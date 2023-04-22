import matplotlib.pyplot as plt
import glob
import argparse
import os
from PerceptualSimilarity import lpips as LPIPS
import sys
from sifid_score import *
from os.path import join
# Specify the directory you want to add
directory_to_add = "/root/autodl-tmp/textual_inversion"
log_dir = '/root/tf-logs'
# Make sure the path is absolute
abs_directory_to_add = os.path.abspath(directory_to_add)

# Add the directory to sys.path
if abs_directory_to_add not in sys.path:
    sys.path.append(abs_directory_to_add)

from tensorboardX import SummaryWriter

def eval_single_category(category, path_gt, fake_anno, step):
    path_pred = join(fake_anno, f'step_{step}')
    ref_file = os.listdir(path_gt)[0]
    files = os.listdir(path_pred)

    tot_dist = 0.
    num_imgs = 0
    for file in files:
        if not os.path.isdir(os.path.join(path_pred,file)):
        # Load images
            # breakpoint()
            img0 = LPIPS.im2tensor(LPIPS.load_image(os.path.join(path_gt,ref_file)))
            img1 = LPIPS.im2tensor(LPIPS.load_image(os.path.join(path_pred,file)))
            if(args.use_gpu):
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0,img1)
            tot_dist += dist01.cpu().item()
            num_imgs += 1

    avg_lpips = tot_dist / num_imgs
    sifid_values = calculate_sifid_ti(path_gt, path_pred, 1, args.gpu != '', 64, 'jpeg')  # 64, 192, 768
    # take average
    sifid_values = np.asarray(sifid_values, dtype=np.float32)
    sifid_mean = sifid_values.mean()

    return avg_lpips, sifid_mean



def save_fig(x, y, xlabel=None, ylabel=None, title=None, output_file="output.png", dpi=300):
    plt.plot(x, y)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')


if __name__ == "__main__":
    # evaluate single step across multiple objects
    # given path2real (gt) and path2real (images of multiple objects parent directory)
    # evaluate the fid, lpips, and mse and return

    """
    For example, path2real: /root/autodl-tmp/textual_inversion/images/(${CATEGORY}_small)
                 path2fake: /root/autodl-tmp/(${CATEGORY}/caption_results/step_${i})
                
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_root', type=str, default='./imgs/ex_dir0')
    parser.add_argument('--fake_root', type=str, default='./imgs/ex_dir1')
    parser.add_argument('-v','--version', type=str, default='0.1')
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--gpu', type=str)

    parser.add_argument('--init_type', type=str, help='experiment name') # caption,
    args = parser.parse_args()

    # select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    categories = ['mug_skulls', 'bowl', 'thin_bird',
                 'car', 'cat', 'clock',
                 'teapot', 'elephant',
                 'backpack', 'glasses', 'sneaker', 'monster']
    # '',

    steps = [i * 20 for i in range(10)] + [i * 100 + 200 for i in range(8)] + [i * 250 + 1000 for i in range(4)] + [i * 500 + 2000 for i in range(7)]

    exp_path = log_dir + f'/{args.init_type}'
    writer = SummaryWriter(exp_path)

    real_root = args.real_root # /root/autodl-tmp/textual_inversion/images
    fake_root = args.fake_root # /root/autodl-tmp

    # eval lpips
    # Initializing the model
    loss_fn = LPIPS.LPIPS(net='alex',version=args.version)
    if(args.use_gpu):
        loss_fn.cuda()

    eval_lpips = True
    eval_sifid = True
    eval_mse = False

    if eval_lpips and eval_sifid:
        lpips_list, sifid_list = [], []
        for step in steps:
            lpips_acc, sifid_acc = 0, 0
            print("Evaluating step: {}".format(step))
            for category in categories:
                print("eval cate {}".format(category))

                real_anno = f'{category}_small'

                pattern = join(fake_root, f"{category}/{args.init_type}_*_results")
                directories = glob.glob(pattern)
                fake_anno = directories[0]

                # mse_anno = fake_anno

                real_path = join(real_root, real_anno)
                avg_lpips, sifid_mean = eval_single_category(category, real_path, fake_anno, step)

                lpips_acc += avg_lpips
                sifid_acc += sifid_mean


            lpips_list.append(lpips_acc / len(categories))
            sifid_list.append(sifid_acc / len(categories))
            writer.add_scalar("LPIPS", lpips_acc / len(categories), step)
            writer.add_scalar("SIFID", sifid_acc / len(categories), step)

        avg_lpips = np.array(lpips_list)
        avg_sifid = np.array(sifid_list)

        save_fig(steps, avg_lpips, title='LPIPS', output_file = 'lpips.png')
        save_fig(steps, avg_sifid, title='SIFID', output_file = 'sifid.png')

    elif eval_mse:
        mse_list = []
        for category in categories:
            pattern = join(fake_root, f"{category}/{args.init_type}_*_init")
            directories = glob.glob(pattern)
            fake_anno = directories[0]
            mse_path = join(fake_anno, 'loss.npy')
            mse = np.load(mse_path)
            mse_list.append(mse)

        # Calculate average mse over categories
        steps = [i+1 for i in range(5000)]
        avg_mse = np.mean(mse_list, axis=0)[:len(steps)]
        save_fig(steps, avg_mse, title='MSE Loss', output_file = 'mse.png')

        # log to tensorboard
        log_dir = "/root/tf-logs/caption_mse_logs"
        writer = SummaryWriter(log_dir)

        for step, mse_value in zip(steps, avg_mse):
            writer.add_scalar('Average MSE', mse_value, step)

#     # Plot average lpips vs steps
#     plot_step = np.array(steps)
#     plt.plot(plot_step, avg_lpips)
#     plt.xlabel("Steps")
#     plt.ylabel("Average LPIPS")
#     plt.title("Average LPIPS over Categories vs Steps")

#     output_file = f'Average_LPIPS_over_{len(categories)}_categories.png'
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')

#     # Plot average mse vs steps
#     plt.plot(plot_step, avg_sifid)
#     plt.xlabel("Steps")
#     plt.ylabel("Average SIFID")
#     plt.title("Average SIFID over Categories vs Steps")
#     # save it
#     output_file = f'Average_SIFID_over_{len(categories)}_categories.png'
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')

#     # Plot average mse vs steps
#     plt.plot(plot_step, avg_mse)
#     plt.xlabel("Steps")
#     plt.ylabel("Average MSE")
#     plt.title("Average MSE over Categories vs Steps")
#     # save it
#     output_file = f'average_over_{len(categories)}_categories.png'
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')


    # read precomputed mse loss
    writer.close()
"""

python eval_images.py --real_root /root/autodl-tmp/textual_inversion/images --fake_root /root/autodl-tmp --gpu 0 --init_type caption

# caption, class, null
"""
