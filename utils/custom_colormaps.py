import numpy as np
import seaborn as sns
import matplotlib as mpl


def create_cmap(color_list, n_colors=None):
    """Create a colormap from a list of colors.

    Parameters
    ----------
    color_list : list
        List of colors in the colormap. Should be a 4-column matrix with
        [:, 0] being the position of each color in the range [0, 1] and
        [:, 1:] being the RGB color values in the range [0, 255]. The
        first color in the list should have position = 0, and the
        last color in the list should have position = 1. Intermediate
        colors should have positions between 0 and 1, listed in
        ascending order. Colors are interpolated between adjacent colors
        in the list. Function has some flexibility to handle color lists
        with only 3 columns (i.e. no color position column), given with
        positions out of order, or with position values or RGB values
        outside the ranges listed, but read the code to understand
        handling before deviating from the recommended formatting.
    n_colors : int
        Number of colors to be included in the final colormap. Must be
        >= the number of colors in color_list. If None, the number of
        colors in color_list is used.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        The final colormap.
    """
    # Ensure color_list is a numpy array.
    color_list = np.asanyarray(color_list)

    # Get n_colors.
    if n_colors is None:
        n_colors = color_list.shape[0]

    # Add a color position column if one is missing.
    if color_list.shape[1] == 3:
        color_list = np.hstack((np.arange(color_list.shape[0])[:, None], color_list))

    # Sort colors in ascending order by position.
    color_list = color_list[np.argsort(color_list[:, 0]), :]

    # Normalize the positions of the colors to the range [0, 1].
    if np.any(color_list[:, 0] > 1) or np.any(color_list[:, 0] < 0):
        _min = np.min(color_list[:, 0])
        _max = np.max(color_list[:, 0])
        color_list[:, 0] = (color_list[:, 0] - _min) / (_max - _min)

    # Normalize the RGB values to the range [0, 1].
    if np.any(color_list[:, 1:] > 1):
        color_list[:, 1:] = color_list[:, 1:] / 255

    # Interpolate between adjacent colors.
    ii = 0
    cmap = []
    for ii in range(color_list.shape[0] - 1):
        cmap += list(
            sns.blend_palette(
                colors=[color_list[ii, 1:], color_list[ii + 1, 1:]],
                n_colors=np.rint(np.diff(color_list[ii : ii + 2, 0])[0] * n_colors),
            )
        )

    # Convert the colormap list to a ListedColormap.
    cmap = mpl.colors.ListedColormap(cmap)

    return cmap


def nih_cmap():
    """Return the NIH colormap as a matplotlib ListedColormap."""
    n_colors = 256
    color_list = np.array(
        [
            [0, 0, 0, 0],
            [0.059, 85, 0, 170],
            [0.122, 0, 0, 85],
            [0.247, 0, 0, 255],
            [0.309, 0, 85, 255],
            [0.372, 0, 170, 170],
            [0.434, 0, 255, 170],
            [0.497, 0, 255, 0],
            [0.559, 85, 255, 85],
            [0.625, 255, 255, 0],
            [0.75, 255, 85, 0],
            [0.85, 255, 0, 0],
            [0.99608, 172, 0, 0],
            [1, 140, 0, 0],
        ]
    )
    cmap = create_cmap(color_list, n_colors)
    return cmap


def avid_cmap():
    """Return the Avid colormap as a matplotlib ListedColormap."""
    n_colors = 255
    color_list = np.array(
        [
            [0/255, 0/255, 0/255],
            [0/255, 2/255, 3/255],
            [0/255, 5/255, 7/255],
            [1/255, 7/255, 10/255],
            [1/255, 9/255, 14/255],
            [1/255, 12/255, 17/255],
            [1/255, 14/255, 20/255],
            [2/255, 16/255, 24/255],
            [2/255, 19/255, 27/255],
            [2/255, 21/255, 30/255],
            [2/255, 23/255, 34/255],
            [2/255, 25/255, 37/255],
            [3/255, 28/255, 41/255],
            [3/255, 30/255, 44/255],
            [3/255, 32/255, 47/255],
            [3/255, 35/255, 51/255],
            [4/255, 37/255, 54/255],
            [4/255, 39/255, 57/255],
            [4/255, 42/255, 61/255],
            [4/255, 44/255, 64/255],
            [4/255, 46/255, 68/255],
            [5/255, 49/255, 71/255],
            [5/255, 51/255, 74/255],
            [5/255, 53/255, 78/255],
            [5/255, 56/255, 81/255],
            [5/255, 58/255, 84/255],
            [6/255, 60/255, 88/255],
            [6/255, 62/255, 91/255],
            [6/255, 65/255, 95/255],
            [6/255, 67/255, 98/255],
            [7/255, 69/255, 101/255],
            [7/255, 72/255, 105/255],
            [7/255, 74/255, 108/255],
            [7/255, 77/255, 108/255],
            [7/255, 79/255, 107/255],
            [7/255, 82/255, 107/255],
            [7/255, 85/255, 107/255],
            [7/255, 87/255, 107/255],
            [8/255, 90/255, 106/255],
            [8/255, 93/255, 106/255],
            [8/255, 95/255, 106/255],
            [8/255, 98/255, 106/255],
            [8/255, 101/255, 105/255],
            [8/255, 103/255, 105/255],
            [8/255, 106/255, 105/255],
            [8/255, 109/255, 105/255],
            [8/255, 111/255, 104/255],
            [8/255, 114/255, 104/255],
            [9/255, 117/255, 104/255],
            [9/255, 120/255, 104/255],
            [9/255, 122/255, 103/255],
            [9/255, 125/255, 103/255],
            [9/255, 128/255, 103/255],
            [9/255, 130/255, 103/255],
            [9/255, 133/255, 102/255],
            [9/255, 136/255, 102/255],
            [9/255, 138/255, 102/255],
            [9/255, 141/255, 102/255],
            [10/255, 144/255, 101/255],
            [10/255, 146/255, 101/255],
            [10/255, 149/255, 101/255],
            [10/255, 152/255, 101/255],
            [10/255, 154/255, 100/255],
            [10/255, 157/255, 100/255],
            [10/255, 157/255, 99/255],
            [11/255, 157/255, 98/255],
            [11/255, 156/255, 97/255],
            [11/255, 156/255, 96/255],
            [12/255, 156/255, 95/255],
            [12/255, 156/255, 95/255],
            [12/255, 156/255, 94/255],
            [13/255, 156/255, 93/255],
            [13/255, 155/255, 92/255],
            [13/255, 155/255, 91/255],
            [14/255, 155/255, 90/255],
            [14/255, 155/255, 89/255],
            [14/255, 155/255, 88/255],
            [15/255, 154/255, 87/255],
            [15/255, 154/255, 86/255],
            [15/255, 154/255, 85/255],
            [16/255, 154/255, 85/255],
            [16/255, 154/255, 84/255],
            [16/255, 154/255, 83/255],
            [17/255, 153/255, 82/255],
            [17/255, 153/255, 81/255],
            [17/255, 153/255, 80/255],
            [18/255, 153/255, 79/255],
            [18/255, 153/255, 78/255],
            [18/255, 152/255, 77/255],
            [19/255, 152/255, 76/255],
            [19/255, 152/255, 75/255],
            [19/255, 152/255, 75/255],
            [20/255, 152/255, 74/255],
            [20/255, 152/255, 73/255],
            [20/255, 151/255, 72/255],
            [21/255, 151/255, 71/255],
            [21/255, 151/255, 70/255],
            [26/255, 147/255, 69/255],
            [31/255, 144/255, 68/255],
            [36/255, 140/255, 67/255],
            [41/255, 137/255, 66/255],
            [46/255, 133/255, 65/255],
            [52/255, 129/255, 64/255],
            [57/255, 126/255, 63/255],
            [62/255, 122/255, 62/255],
            [67/255, 118/255, 60/255],
            [72/255, 115/255, 59/255],
            [77/255, 111/255, 58/255],
            [82/255, 108/255, 57/255],
            [87/255, 104/255, 56/255],
            [92/255, 100/255, 55/255],
            [97/255, 97/255, 54/255],
            [103/255, 93/255, 53/255],
            [108/255, 89/255, 52/255],
            [113/255, 86/255, 51/255],
            [118/255, 82/255, 50/255],
            [123/255, 79/255, 49/255],
            [128/255, 75/255, 48/255],
            [133/255, 71/255, 47/255],
            [138/255, 68/255, 46/255],
            [143/255, 64/255, 45/255],
            [148/255, 60/255, 43/255],
            [153/255, 57/255, 42/255],
            [159/255, 53/255, 41/255],
            [164/255, 50/255, 40/255],
            [169/255, 46/255, 39/255],
            [174/255, 42/255, 38/255],
            [179/255, 39/255, 37/255],
            [184/255, 35/255, 36/255],
            [185/255, 36/255, 36/255],
            [186/255, 37/255, 36/255],
            [187/255, 38/255, 36/255],
            [188/255, 39/255, 36/255],
            [189/255, 40/255, 36/255],
            [189/255, 41/255, 36/255],
            [190/255, 42/255, 36/255],
            [191/255, 43/255, 36/255],
            [192/255, 43/255, 37/255],
            [193/255, 44/255, 37/255],
            [194/255, 45/255, 37/255],
            [195/255, 46/255, 37/255],
            [196/255, 47/255, 37/255],
            [197/255, 48/255, 37/255],
            [198/255, 49/255, 37/255],
            [199/255, 50/255, 37/255],
            [199/255, 51/255, 37/255],
            [200/255, 52/255, 37/255],
            [201/255, 53/255, 37/255],
            [202/255, 54/255, 37/255],
            [203/255, 55/255, 37/255],
            [204/255, 56/255, 37/255],
            [205/255, 57/255, 37/255],
            [206/255, 58/255, 37/255],
            [207/255, 58/255, 38/255],
            [208/255, 59/255, 38/255],
            [209/255, 60/255, 38/255],
            [209/255, 61/255, 38/255],
            [210/255, 62/255, 38/255],
            [211/255, 63/255, 38/255],
            [212/255, 64/255, 38/255],
            [213/255, 65/255, 38/255],
            [214/255, 66/255, 38/255],
            [215/255, 68/255, 38/255],
            [215/255, 71/255, 38/255],
            [216/255, 73/255, 38/255],
            [216/255, 75/255, 38/255],
            [217/255, 77/255, 38/255],
            [217/255, 80/255, 37/255],
            [218/255, 82/255, 37/255],
            [218/255, 84/255, 37/255],
            [219/255, 86/255, 37/255],
            [219/255, 89/255, 37/255],
            [220/255, 91/255, 37/255],
            [220/255, 93/255, 37/255],
            [221/255, 95/255, 37/255],
            [221/255, 98/255, 37/255],
            [222/255, 100/255, 37/255],
            [222/255, 102/255, 36/255],
            [223/255, 104/255, 36/255],
            [223/255, 107/255, 36/255],
            [224/255, 109/255, 36/255],
            [224/255, 111/255, 36/255],
            [225/255, 113/255, 36/255],
            [225/255, 116/255, 36/255],
            [226/255, 118/255, 36/255],
            [226/255, 120/255, 36/255],
            [227/255, 122/255, 36/255],
            [227/255, 125/255, 35/255],
            [228/255, 127/255, 35/255],
            [228/255, 129/255, 35/255],
            [229/255, 131/255, 35/255],
            [229/255, 134/255, 35/255],
            [230/255, 136/255, 35/255],
            [230/255, 138/255, 34/255],
            [231/255, 140/255, 34/255],
            [231/255, 142/255, 33/255],
            [231/255, 144/255, 33/255],
            [232/255, 146/255, 32/255],
            [232/255, 148/255, 32/255],
            [233/255, 150/255, 31/255],
            [233/255, 152/255, 30/255],
            [233/255, 154/255, 30/255],
            [234/255, 156/255, 29/255],
            [234/255, 158/255, 29/255],
            [234/255, 160/255, 28/255],
            [235/255, 162/255, 28/255],
            [235/255, 164/255, 27/255],
            [235/255, 166/255, 26/255],
            [236/255, 168/255, 26/255],
            [236/255, 169/255, 25/255],
            [237/255, 171/255, 25/255],
            [237/255, 173/255, 24/255],
            [237/255, 175/255, 23/255],
            [238/255, 177/255, 23/255],
            [238/255, 179/255, 22/255],
            [238/255, 181/255, 22/255],
            [239/255, 183/255, 21/255],
            [239/255, 185/255, 21/255],
            [239/255, 187/255, 20/255],
            [240/255, 189/255, 19/255],
            [240/255, 191/255, 19/255],
            [241/255, 193/255, 18/255],
            [241/255, 195/255, 18/255],
            [241/255, 197/255, 17/255],
            [242/255, 199/255, 17/255],
            [242/255, 201/255, 16/255],
            [242/255, 202/255, 16/255],
            [242/255, 203/255, 16/255],
            [242/255, 204/255, 17/255],
            [242/255, 205/255, 17/255],
            [241/255, 207/255, 17/255],
            [241/255, 208/255, 17/255],
            [241/255, 209/255, 18/255],
            [241/255, 210/255, 18/255],
            [241/255, 211/255, 18/255],
            [241/255, 212/255, 18/255],
            [241/255, 213/255, 19/255],
            [241/255, 214/255, 19/255],
            [241/255, 215/255, 19/255],
            [241/255, 216/255, 19/255],
            [240/255, 218/255, 20/255],
            [240/255, 219/255, 20/255],
            [240/255, 220/255, 20/255],
            [240/255, 221/255, 20/255],
            [240/255, 222/255, 21/255],
            [240/255, 223/255, 21/255],
            [240/255, 224/255, 21/255],
            [240/255, 225/255, 21/255],
            [240/255, 226/255, 22/255],
            [240/255, 227/255, 22/255],
            [239/255, 229/255, 22/255],
            [239/255, 230/255, 22/255],
            [239/255, 231/255, 23/255],
            [239/255, 232/255, 23/255],
            [239/255, 233/255, 23/255],
        ]
    )
    cmap = create_cmap(color_list, n_colors)
    return cmap

def tauvid1_cmap():
    color_list = np.array(
        [
            [0,	0,	0,],
            [0.00818181876093149, 0.0218181814998388, 0.0381818152964115],
            [0.0163636375218630, 0.0436363629996777, 0.0763636305928230],
            [0.0245454553514719, 0.0654545426368713, 0.114545449614525],
            [0.0327272750437260, 0.0872727259993553, 0.152727261185646],
            [0.0409090928733349, 0.109090909361839, 0.190909087657928],
            [0.0490909107029438, 0.130909085273743,	0.229090899229050],
            [0.0572727285325527, 0.152727276086807,	0.267272710800171],
            [0.0654545500874519, 0.174545451998711,	0.305454522371292],
            [0.0736363679170609, 0.196363627910614,	0.343636363744736],
            [0.0818181857466698, 0.218181818723679,	0.381818175315857],
            [0.0900000035762787, 0.239999994635582,	0.419999986886978],
            [0.0927272737026215, 0.273636370897293,	0.409090906381607],
            [0.0954545512795448, 0.307272732257843,	0.398181796073914],
            [0.0981818214058876, 0.340909093618393,	0.387272715568543],
            [0.100909091532230,	0.374545454978943, 0.376363635063171],
            [0.103636361658573,	0.408181816339493, 0.365454554557800],
            [0.106363639235497,	0.441818177700043, 0.354545444250107],
            [0.109090909361839,	0.475454539060593, 0.343636363744736],
            [0.111818179488182,	0.509090900421143, 0.332727283239365],
            [0.114545449614525,	0.542727291584015, 0.321818202733994],
            [0.117272727191448,	0.576363623142242, 0.310909092426300],
            [0.119999997317791,	0.610000014305115, 0.300000011920929],
            [0.170000001788139,	0.586250007152557, 0.285000026226044],
            [0.219999998807907,	0.562500000000000, 0.270000010728836],
            [0.269999980926514,	0.538749992847443, 0.254999995231628],
            [0.319999992847443,	0.514999985694885, 0.240000009536743],
            [0.369999974966049,	0.491250008344650, 0.225000008940697],
            [0.419999986886978,	0.467500001192093, 0.210000008344650],
            [0.469999969005585,	0.443749994039536, 0.195000007748604],
            [0.519999980926514,	0.419999986886978, 0.180000007152557],
            [0.550000011920929,	0.389999985694885, 0.180000007152557],
            [0.580000042915344,	0.359999984502792, 0.177500009536743],
            [0.610000014305115,	0.329999983310699, 0.175000011920929],
            [0.639999985694885,	0.299999982118607, 0.172499999403954],
            [0.670000016689301,	0.269999980926514, 0.170000001788139],
            [0.700000047683716,	0.239999994635582, 0.167500004172325],
            [0.730000019073486,	0.210000008344650, 0.164999991655350],
            [0.759999990463257,	0.180000007152557, 0.162499994039536],
            [0.790000021457672,	0.150000005960464, 0.159999996423721],
            [0.798333346843720,	0.180000007152557, 0.164999991655350],
            [0.806666672229767,	0.210000008344650, 0.170000001788139],
            [0.814999997615814,	0.240000009536743, 0.174999997019768],
            [0.823333323001862,	0.270000010728836, 0.179999992251396],
            [0.831666648387909,	0.300000011920929, 0.185000002384186],
            [0.840000033378601,	0.329999983310699, 0.189999997615814],
            [0.848333358764648,	0.359999984502792, 0.194999992847443],
            [0.856666684150696,	0.389999985694885, 0.200000002980232],
            [0.865000009536743,	0.419999986886978, 0.204999998211861],
            [0.873333334922791,	0.449999988079071, 0.209999993443489],
            [0.881666660308838,	0.479999989271164, 0.215000003576279],
            [0.889999985694885,	0.509999990463257, 0.219999998807907],
            [0.897241652011871,	0.549491643905640, 0.208374992012978],
            [0.904483318328857,	0.588983297348023, 0.196750000119209],
            [0.911724984645844,	0.628475010395050, 0.185124993324280],
            [0.918966650962830,	0.667966663837433, 0.173500001430511],
            [0.926208317279816,	0.707458317279816, 0.161874994635582],
            [0.933449983596802,	0.746950030326843, 0.150250002741814],
            [0.940691649913788,	0.786441683769226, 0.138624995946884],
            [0.947933316230774,	0.825933337211609, 0.127000004053116],
            [0.955174982547760,	0.865424990653992, 0.115374997258186],
            [0.962416648864746,	0.904916644096375, 0.103749997913837],
            [0.969658315181732,	0.944408357143402, 0.0921249985694885],
            [0.976899981498718,	0.983900010585785, 0.0804999992251396],
        ]
    )
    cmap = create_cmap(color_list)
    return cmap

def turbo_cmap():
    color_list = np.array(
        [
            [0.18995, 0.07176, 0.23217],
            [0.19483, 0.08339, 0.26149],
            [0.19956, 0.09498, 0.29024],
            [0.20415, 0.10652, 0.31844],
            [0.20860, 0.11802, 0.34607],
            [0.21291, 0.12947, 0.37314],
            [0.21708, 0.14087, 0.39964],
            [0.22111, 0.15223, 0.42558],
            [0.22500, 0.16354, 0.45096],
            [0.22875, 0.17481, 0.47578],
            [0.23236, 0.18603, 0.50004],
            [0.23582, 0.19720, 0.52373],
            [0.23915, 0.20833, 0.54686],
            [0.24234, 0.21941, 0.56942],
            [0.24539, 0.23044, 0.59142],
            [0.24830, 0.24143, 0.61286],
            [0.25107, 0.25237, 0.63374],
            [0.25369, 0.26327, 0.65406],
            [0.25618, 0.27412, 0.67381],
            [0.25853, 0.28492, 0.69300],
            [0.26074, 0.29568, 0.71162],
            [0.26280, 0.30639, 0.72968],
            [0.26473, 0.31706, 0.74718],
            [0.26652, 0.32768, 0.76412],
            [0.26816, 0.33825, 0.78050],
            [0.26967, 0.34878, 0.79631],
            [0.27103, 0.35926, 0.81156],
            [0.27226, 0.36970, 0.82624],
            [0.27334, 0.38008, 0.84037],
            [0.27429, 0.39043, 0.85393],
            [0.27509, 0.40072, 0.86692],
            [0.27576, 0.41097, 0.87936],
            [0.27628, 0.42118, 0.89123],
            [0.27667, 0.43134, 0.90254],
            [0.27691, 0.44145, 0.91328],
            [0.27701, 0.45152, 0.92347],
            [0.27698, 0.46153, 0.93309],
            [0.27680, 0.47151, 0.94214],
            [0.27648, 0.48144, 0.95064],
            [0.27603, 0.49132, 0.95857],
            [0.27543, 0.50115, 0.96594],
            [0.27469, 0.51094, 0.97275],
            [0.27381, 0.52069, 0.97899],
            [0.27273, 0.53040, 0.98461],
            [0.27106, 0.54015, 0.98930],
            [0.26878, 0.54995, 0.99303],
            [0.26592, 0.55979, 0.99583],
            [0.26252, 0.56967, 0.99773],
            [0.25862, 0.57958, 0.99876],
            [0.25425, 0.58950, 0.99896],
            [0.24946, 0.59943, 0.99835],
            [0.24427, 0.60937, 0.99697],
            [0.23874, 0.61931, 0.99485],
            [0.23288, 0.62923, 0.99202],
            [0.22676, 0.63913, 0.98851],
            [0.22039, 0.64901, 0.98436],
            [0.21382, 0.65886, 0.97959],
            [0.20708, 0.66866, 0.97423],
            [0.20021, 0.67842, 0.96833],
            [0.19326, 0.68812, 0.96190],
            [0.18625, 0.69775, 0.95498],
            [0.17923, 0.70732, 0.94761],
            [0.17223, 0.71680, 0.93981],
            [0.16529, 0.72620, 0.93161],
            [0.15844, 0.73551, 0.92305],
            [0.15173, 0.74472, 0.91416],
            [0.14519, 0.75381, 0.90496],
            [0.13886, 0.76279, 0.89550],
            [0.13278, 0.77165, 0.88580],
            [0.12698, 0.78037, 0.87590],
            [0.12151, 0.78896, 0.86581],
            [0.11639, 0.79740, 0.85559],
            [0.11167, 0.80569, 0.84525],
            [0.10738, 0.81381, 0.83484],
            [0.10357, 0.82177, 0.82437],
            [0.10026, 0.82955, 0.81389],
            [0.09750, 0.83714, 0.80342],
            [0.09532, 0.84455, 0.79299],
            [0.09377, 0.85175, 0.78264],
            [0.09287, 0.85875, 0.77240],
            [0.09267, 0.86554, 0.76230],
            [0.09320, 0.87211, 0.75237],
            [0.09451, 0.87844, 0.74265],
            [0.09662, 0.88454, 0.73316],
            [0.09958, 0.89040, 0.72393],
            [0.10342, 0.89600, 0.71500],
            [0.10815, 0.90142, 0.70599],
            [0.11374, 0.90673, 0.69651],
            [0.12014, 0.91193, 0.68660],
            [0.12733, 0.91701, 0.67627],
            [0.13526, 0.92197, 0.66556],
            [0.14391, 0.92680, 0.65448],
            [0.15323, 0.93151, 0.64308],
            [0.16319, 0.93609, 0.63137],
            [0.17377, 0.94053, 0.61938],
            [0.18491, 0.94484, 0.60713],
            [0.19659, 0.94901, 0.59466],
            [0.20877, 0.95304, 0.58199],
            [0.22142, 0.95692, 0.56914],
            [0.23449, 0.96065, 0.55614],
            [0.24797, 0.96423, 0.54303],
            [0.26180, 0.96765, 0.52981],
            [0.27597, 0.97092, 0.51653],
            [0.29042, 0.97403, 0.50321],
            [0.30513, 0.97697, 0.48987],
            [0.32006, 0.97974, 0.47654],
            [0.33517, 0.98234, 0.46325],
            [0.35043, 0.98477, 0.45002],
            [0.36581, 0.98702, 0.43688],
            [0.38127, 0.98909, 0.42386],
            [0.39678, 0.99098, 0.41098],
            [0.41229, 0.99268, 0.39826],
            [0.42778, 0.99419, 0.38575],
            [0.44321, 0.99551, 0.37345],
            [0.45854, 0.99663, 0.36140],
            [0.47375, 0.99755, 0.34963],
            [0.48879, 0.99828, 0.33816],
            [0.50362, 0.99879, 0.32701],
            [0.51822, 0.99910, 0.31622],
            [0.53255, 0.99919, 0.30581],
            [0.54658, 0.99907, 0.29581],
            [0.56026, 0.99873, 0.28623],
            [0.57357, 0.99817, 0.27712],
            [0.58646, 0.99739, 0.26849],
            [0.59891, 0.99638, 0.26038],
            [0.61088, 0.99514, 0.25280],
            [0.62233, 0.99366, 0.24579],
            [0.63323, 0.99195, 0.23937],
            [0.64362, 0.98999, 0.23356],
            [0.65394, 0.98775, 0.22835],
            [0.66428, 0.98524, 0.22370],
            [0.67462, 0.98246, 0.21960],
            [0.68494, 0.97941, 0.21602],
            [0.69525, 0.97610, 0.21294],
            [0.70553, 0.97255, 0.21032],
            [0.71577, 0.96875, 0.20815],
            [0.72596, 0.96470, 0.20640],
            [0.73610, 0.96043, 0.20504],
            [0.74617, 0.95593, 0.20406],
            [0.75617, 0.95121, 0.20343],
            [0.76608, 0.94627, 0.20311],
            [0.77591, 0.94113, 0.20310],
            [0.78563, 0.93579, 0.20336],
            [0.79524, 0.93025, 0.20386],
            [0.80473, 0.92452, 0.20459],
            [0.81410, 0.91861, 0.20552],
            [0.82333, 0.91253, 0.20663],
            [0.83241, 0.90627, 0.20788],
            [0.84133, 0.89986, 0.20926],
            [0.85010, 0.89328, 0.21074],
            [0.85868, 0.88655, 0.21230],
            [0.86709, 0.87968, 0.21391],
            [0.87530, 0.87267, 0.21555],
            [0.88331, 0.86553, 0.21719],
            [0.89112, 0.85826, 0.21880],
            [0.89870, 0.85087, 0.22038],
            [0.90605, 0.84337, 0.22188],
            [0.91317, 0.83576, 0.22328],
            [0.92004, 0.82806, 0.22456],
            [0.92666, 0.82025, 0.22570],
            [0.93301, 0.81236, 0.22667],
            [0.93909, 0.80439, 0.22744],
            [0.94489, 0.79634, 0.22800],
            [0.95039, 0.78823, 0.22831],
            [0.95560, 0.78005, 0.22836],
            [0.96049, 0.77181, 0.22811],
            [0.96507, 0.76352, 0.22754],
            [0.96931, 0.75519, 0.22663],
            [0.97323, 0.74682, 0.22536],
            [0.97679, 0.73842, 0.22369],
            [0.98000, 0.73000, 0.22161],
            [0.98289, 0.72140, 0.21918],
            [0.98549, 0.71250, 0.21650],
            [0.98781, 0.70330, 0.21358],
            [0.98986, 0.69382, 0.21043],
            [0.99163, 0.68408, 0.20706],
            [0.99314, 0.67408, 0.20348],
            [0.99438, 0.66386, 0.19971],
            [0.99535, 0.65341, 0.19577],
            [0.99607, 0.64277, 0.19165],
            [0.99654, 0.63193, 0.18738],
            [0.99675, 0.62093, 0.18297],
            [0.99672, 0.60977, 0.17842],
            [0.99644, 0.59846, 0.17376],
            [0.99593, 0.58703, 0.16899],
            [0.99517, 0.57549, 0.16412],
            [0.99419, 0.56386, 0.15918],
            [0.99297, 0.55214, 0.15417],
            [0.99153, 0.54036, 0.14910],
            [0.98987, 0.52854, 0.14398],
            [0.98799, 0.51667, 0.13883],
            [0.98590, 0.50479, 0.13367],
            [0.98360, 0.49291, 0.12849],
            [0.98108, 0.48104, 0.12332],
            [0.97837, 0.46920, 0.11817],
            [0.97545, 0.45740, 0.11305],
            [0.97234, 0.44565, 0.10797],
            [0.96904, 0.43399, 0.10294],
            [0.96555, 0.42241, 0.09798],
            [0.96187, 0.41093, 0.09310],
            [0.95801, 0.39958, 0.08831],
            [0.95398, 0.38836, 0.08362],
            [0.94977, 0.37729, 0.07905],
            [0.94538, 0.36638, 0.07461],
            [0.94084, 0.35566, 0.07031],
            [0.93612, 0.34513, 0.06616],
            [0.93125, 0.33482, 0.06218],
            [0.92623, 0.32473, 0.05837],
            [0.92105, 0.31489, 0.05475],
            [0.91572, 0.30530, 0.05134],
            [0.91024, 0.29599, 0.04814],
            [0.90463, 0.28696, 0.04516],
            [0.89888, 0.27824, 0.04243],
            [0.89298, 0.26981, 0.03993],
            [0.88691, 0.26152, 0.03753],
            [0.88066, 0.25334, 0.03521],
            [0.87422, 0.24526, 0.03297],
            [0.86760, 0.23730, 0.03082],
            [0.86079, 0.22945, 0.02875],
            [0.85380, 0.22170, 0.02677],
            [0.84662, 0.21407, 0.02487],
            [0.83926, 0.20654, 0.02305],
            [0.83172, 0.19912, 0.02131],
            [0.82399, 0.19182, 0.01966],
            [0.81608, 0.18462, 0.01809],
            [0.80799, 0.17753, 0.01660],
            [0.79971, 0.17055, 0.01520],
            [0.79125, 0.16368, 0.01387],
            [0.78260, 0.15693, 0.01264],
            [0.77377, 0.15028, 0.01148],
            [0.76476, 0.14374, 0.01041],
            [0.75556, 0.13731, 0.00942],
            [0.74617, 0.13098, 0.00851],
            [0.73661, 0.12477, 0.00769],
            [0.72686, 0.11867, 0.00695],
            [0.71692, 0.11268, 0.00629],
            [0.70680, 0.10680, 0.00571],
            [0.69650, 0.10102, 0.00522],
            [0.68602, 0.09536, 0.00481],
            [0.67535, 0.08980, 0.00449],
            [0.66449, 0.08436, 0.00424],
            [0.65345, 0.07902, 0.00408],
            [0.64223, 0.07380, 0.00401],
            [0.63082, 0.06868, 0.00401],
            [0.61923, 0.06367, 0.00410],
            [0.60746, 0.05878, 0.00427],
            [0.59550, 0.05399, 0.00453],
            [0.58336, 0.04931, 0.00486],
            [0.57103, 0.04474, 0.00529],
            [0.55852, 0.04028, 0.00579],
            [0.54583, 0.03593, 0.00638],
            [0.53295, 0.03169, 0.00705],
            [0.51989, 0.02756, 0.00780],
            [0.50664, 0.02354, 0.00863],
            [0.49321, 0.01963, 0.00955],
            [0.47960, 0.01583, 0.01055],
        ]
    )
    cmap = create_cmap(color_list)
    return cmap
