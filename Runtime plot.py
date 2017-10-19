import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.ticker import MultipleLocator

mnist_conj_rt=[ 52.23479104,52.0833478, 52.13528895,52.16887331,52.14740992
,51.93755293,52.05513811,52.18399668,52.14517093,52.11357617
,52.21507597,52.13511062,52.12233639,52.07747984,52.15827227
,52.13944793,52.23459196,52.03342485,52.17125177,52.06109118
,52.0510664, 52.03592944,51.90785313,52.04991961,52.19029593
,52.10212922,52.00649428,51.93921518,51.99659801]


mnist_alt_rt=[66.5880723, 66.34301877,66.57836866,66.53362823,66.51431751
,66.49570084,66.43020725,66.49817848,66.35477948,66.26882815
,66.44768858,66.56072903,66.43712592,66.36190057,66.46439004
,66.44675994,66.49743438,66.52100563,66.530267,,66.48819327
,66.46549129,66.40073872,66.5257442, 66.43581247,66.44559813
,66.45740032,66.63413739,66.58587122,66.56617188]

cifar_alt_rt=[ 49.35915947,49.36687732,49.28288412,49.18178153,49.17465615
,49.18597007,49.2421658, 49.14124608,49.17512965,49.11450267
,49.09186578,49.10301518,49.11930323,49.1852181, 49.1588738
,49.14324474,49.24778938,49.21907949,49.18965125,49.24055028
,49.16151786,49.18580103,49.12525988,49.18915176,49.1377449
,49.12484121,49.08380938,49.02378058,48.93323255,49.033705
,49.01467609,48.9785583, 48.93765521,48.9334209, 48.91179013
,48.91555285,48.87340641,48.84589505,48.83789682,48.93482399
,48.87183619,48.90121198,48.85315776,48.86218309,48.84932661
,48.82244492,48.89725089,48.8457222, 48.92619395]
