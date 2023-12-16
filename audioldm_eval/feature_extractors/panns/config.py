import numpy as np
import csv

sample_rate = 32000
clip_samples = sample_rate * 10  # Audio clips are 10-second

# Load label
with open("metadata/class_labels_indices.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    lines = list(reader)

labels = []
ids = []  # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

classes_num = len(labels)

lb_to_ix = {label: i for i, label in enumerate(labels)}
ix_to_lb = {i: label for i, label in enumerate(labels)}

id_to_ix = {id: i for i, id in enumerate(ids)}
ix_to_id = {i: id for i, id in enumerate(ids)}

full_samples_per_class = np.array(
    [
        937432,
        16344,
        7822,
        10271,
        2043,
        14420,
        733,
        1511,
        1258,
        424,
        1751,
        704,
        369,
        590,
        1063,
        1375,
        5026,
        743,
        853,
        1648,
        714,
        1497,
        1251,
        2139,
        1093,
        133,
        224,
        39469,
        6423,
        407,
        1559,
        4546,
        6826,
        7464,
        2468,
        549,
        4063,
        334,
        587,
        238,
        1766,
        691,
        114,
        2153,
        236,
        209,
        421,
        740,
        269,
        959,
        137,
        4192,
        485,
        1515,
        655,
        274,
        69,
        157,
        1128,
        807,
        1022,
        346,
        98,
        680,
        890,
        352,
        4169,
        2061,
        1753,
        9883,
        1339,
        708,
        37857,
        18504,
        12864,
        2475,
        2182,
        757,
        3624,
        677,
        1683,
        3583,
        444,
        1780,
        2364,
        409,
        4060,
        3097,
        3143,
        502,
        723,
        600,
        230,
        852,
        1498,
        1865,
        1879,
        2429,
        5498,
        5430,
        2139,
        1761,
        1051,
        831,
        2401,
        2258,
        1672,
        1711,
        987,
        646,
        794,
        25061,
        5792,
        4256,
        96,
        8126,
        2740,
        752,
        513,
        554,
        106,
        254,
        1592,
        556,
        331,
        615,
        2841,
        737,
        265,
        1349,
        358,
        1731,
        1115,
        295,
        1070,
        972,
        174,
        937780,
        112337,
        42509,
        49200,
        11415,
        6092,
        13851,
        2665,
        1678,
        13344,
        2329,
        1415,
        2244,
        1099,
        5024,
        9872,
        10948,
        4409,
        2732,
        1211,
        1289,
        4807,
        5136,
        1867,
        16134,
        14519,
        3086,
        19261,
        6499,
        4273,
        2790,
        8820,
        1228,
        1575,
        4420,
        3685,
        2019,
        664,
        324,
        513,
        411,
        436,
        2997,
        5162,
        3806,
        1389,
        899,
        8088,
        7004,
        1105,
        3633,
        2621,
        9753,
        1082,
        26854,
        3415,
        4991,
        2129,
        5546,
        4489,
        2850,
        1977,
        1908,
        1719,
        1106,
        1049,
        152,
        136,
        802,
        488,
        592,
        2081,
        2712,
        1665,
        1128,
        250,
        544,
        789,
        2715,
        8063,
        7056,
        2267,
        8034,
        6092,
        3815,
        1833,
        3277,
        8813,
        2111,
        4662,
        2678,
        2954,
        5227,
        1472,
        2591,
        3714,
        1974,
        1795,
        4680,
        3751,
        6585,
        2109,
        36617,
        6083,
        16264,
        17351,
        3449,
        5034,
        3931,
        2599,
        4134,
        3892,
        2334,
        2211,
        4516,
        2766,
        2862,
        3422,
        1788,
        2544,
        2403,
        2892,
        4042,
        3460,
        1516,
        1972,
        1563,
        1579,
        2776,
        1647,
        4535,
        3921,
        1261,
        6074,
        2922,
        3068,
        1948,
        4407,
        712,
        1294,
        1019,
        1572,
        3764,
        5218,
        975,
        1539,
        6376,
        1606,
        6091,
        1138,
        1169,
        7925,
        3136,
        1108,
        2677,
        2680,
        1383,
        3144,
        2653,
        1986,
        1800,
        1308,
        1344,
        122231,
        12977,
        2552,
        2678,
        7824,
        768,
        8587,
        39503,
        3474,
        661,
        430,
        193,
        1405,
        1442,
        3588,
        6280,
        10515,
        785,
        710,
        305,
        206,
        4990,
        5329,
        3398,
        1771,
        3022,
        6907,
        1523,
        8588,
        12203,
        666,
        2113,
        7916,
        434,
        1636,
        5185,
        1062,
        664,
        952,
        3490,
        2811,
        2749,
        2848,
        15555,
        363,
        117,
        1494,
        1647,
        5886,
        4021,
        633,
        1013,
        5951,
        11343,
        2324,
        243,
        372,
        943,
        734,
        242,
        3161,
        122,
        127,
        201,
        1654,
        768,
        134,
        1467,
        642,
        1148,
        2156,
        1368,
        1176,
        302,
        1909,
        61,
        223,
        1812,
        287,
        422,
        311,
        228,
        748,
        230,
        1876,
        539,
        1814,
        737,
        689,
        1140,
        591,
        943,
        353,
        289,
        198,
        490,
        7938,
        1841,
        850,
        457,
        814,
        146,
        551,
        728,
        1627,
        620,
        648,
        1621,
        2731,
        535,
        88,
        1736,
        736,
        328,
        293,
        3170,
        344,
        384,
        7640,
        433,
        215,
        715,
        626,
        128,
        3059,
        1833,
        2069,
        3732,
        1640,
        1508,
        836,
        567,
        2837,
        1151,
        2068,
        695,
        1494,
        3173,
        364,
        88,
        188,
        740,
        677,
        273,
        1533,
        821,
        1091,
        293,
        647,
        318,
        1202,
        328,
        532,
        2847,
        526,
        721,
        370,
        258,
        956,
        1269,
        1641,
        339,
        1322,
        4485,
        286,
        1874,
        277,
        757,
        1393,
        1330,
        380,
        146,
        377,
        394,
        318,
        339,
        1477,
        1886,
        101,
        1435,
        284,
        1425,
        686,
        621,
        221,
        117,
        87,
        1340,
        201,
        1243,
        1222,
        651,
        1899,
        421,
        712,
        1016,
        1279,
        124,
        351,
        258,
        7043,
        368,
        666,
        162,
        7664,
        137,
        70159,
        26179,
        6321,
        32236,
        33320,
        771,
        1169,
        269,
        1103,
        444,
        364,
        2710,
        121,
        751,
        1609,
        855,
        1141,
        2287,
        1940,
        3943,
        289,
    ]
)
