import copy
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mpl_scatter_density
import numpy as np
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import image
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


def muti_draw_picture(NDVI, predict, save_name):
    # Density Map
    # set the Chinese font type
    plt.rcParams['font.sans-serif'] = ["DFKai-SB"]
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(5.55, 4.2))  # set the figure and the size
    density_map = fig.add_subplot(111, projection='scatter_density')  # Add a subplot in the figure
    # density_map.set_title("點密度圖",size=20) #set the title
    normalize = ImageNormalize(vmin=0, vmax=100, stretch=LogStretch())
    density = density_map.scatter_density(predict, NDVI, cmap=plt.cm.Blues, norm=normalize)
    density_map.plot([-1, 1], [-1, 1], 'red')
    density_map.set_xlim(-1, 1)
    density_map.set_ylim(-1, 1)
    ticks_array = np.linspace(-1, 1, 9)
    density_map.set_xticks(ticks_array)
    density_map.set_yticks(ticks_array)
    density_map.set_xticklabels(ticks_array)
    density_map.set_yticklabels(ticks_array)
    density_map.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    density_map.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # divider = make_axes_locatable(density_map)
    # cax = divider.append_axes('right', size='5%', pad=0.1)
    colorbar = plt.colorbar(density)  # , cax=cax, orientation='vertical')
    colorbar.set_ticks(np.linspace(0, 100, 6))
    colorbar.set_ticklabels([0, 20, 40, 60, 80, 100])
    # colorbar.ax.tick_params(labelsize=50)
    fig.tight_layout()
    fig.savefig(Path(f"outputs/{save_name}_dmap.jpg"))
    # plt.show()
    plt.close(fig)


def vi_func(name, NDVI, RGB, VI, formula, bound):
    R, G, B = [RGB[:, :, band] for band in range(3)]

    if VI is None:
        with np.errstate(invalid='ignore'):
            VI = formula(R, G, B)

    NDVI_s = copy.deepcopy(NDVI).reshape(-1)
    VI_s = copy.deepcopy(VI).reshape(-1)

    NDVI_s = NDVI_s[~(np.isnan(VI_s) | np.isinf(VI_s))]
    VI_s = VI_s[~(np.isnan(VI_s) | np.isinf(VI_s))]

    vmin = vmax = None

    if bound is None:
        vmin, vmax = np.min(VI_s), np.max(VI_s)
    else:
        vmin, vmax = bound

    VI_s[VI_s < vmin] = vmin
    VI_s[VI_s > vmax] = vmax

    VI_cor = stats.pearsonr(NDVI_s, VI_s)[0]
    # VI_r2 = r2_score(NDVI_s, VI_s)

    print(f"{name:6} cor: {VI_cor:.3f}")
    # print(f"{name:6} r2 : {VI_r2:.3f}")

    cmap = plt.get_cmap("jet")
    cmap.set_bad("white")

    if name == "CNN":
        image.imsave(Path(f"outputs/{name}.jpg"), VI, vmin=vmin, vmax=vmax, cmap=cmap)
        muti_draw_picture(NDVI_s, VI_s, name)
    else:
        image.imsave(Path(f"outputs/{name}.jpg"), VI, vmin=vmin, vmax=vmax, cmap=cmap)
        mm_scaler = MinMaxScaler((-1, 1))
        VI_norm_s = mm_scaler.fit_transform(VI_s.reshape(-1, 1)).reshape(-1)
        muti_draw_picture(NDVI_s, VI_norm_s, name)


if __name__ == "__main__":
    rgb_path = Path("../Jim/dataset/testing/test_15meter_RGB_f32.npy")
    ndvi_path = Path("../Jim/dataset/testing/test_15meter_NDVI_f32.npy")
    vi_path = Path("outputs/weight_199_15m/predict.npy")

    rgb_img = np.load(rgb_path, mmap_mode='r', allow_pickle=True)
    NDVI = np.load(ndvi_path, mmap_mode='r', allow_pickle=True)
    VI = np.load(vi_path, mmap_mode='r', allow_pickle=True)

    # 1 Soil and Greenhouse
    image_index = 164
    NDVI = NDVI[image_index, :, :]
    rgb_img = rgb_img[image_index, :, :, :]
    VI = VI[image_index, :, :, 0]

    # 2 Water
    # image_index = 257
    # NDVI = NDVI[image_index, :-53, :-20]
    # rgb_img = rgb_img[image_index, :-53, :-20, :]
    # VI = VI[image_index, :-53, :-20, 0]

    # 3 Forest
    # image_index = 283
    # NDVI = NDVI[image_index, :-53, :-20]
    # rgb_img = rgb_img[image_index, :-53, :-20, :]
    # VI = VI[image_index, :-53, :-20, 0]

    image.imsave(Path(f"outputs/RGB.jpg"), rgb_img)
    image.imsave(Path(f"outputs/NDVI.jpg"), np.squeeze(NDVI), vmin=-1, vmax=1, cmap=plt.get_cmap("jet"))

    vi_list = [
        ("CNN", None, VI, (-1, 1)),
        ("GRVI", lambda R, G, B: (G - R) / (G + R), None, (-1, 1)),
        ("ExG", lambda R, G, B: 2 * G - R - B, None, None),
        ("TGI", lambda R, G, B: -0.39 * R + G - 0.61 * B, None, None),
        ("VARI", lambda R, G, B: (G - R) / (G + R - B), None, (-0.75, 0.75)),
    ]

    for name, formula, vi, bound in vi_list:
        vi_func(name, NDVI, rgb_img, vi, formula, bound)
