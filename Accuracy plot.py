import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.ticker import MultipleLocator

acc_con_mnist=[0.952224,0.976763,0.982772,0.985677,0.987981,0.990284,0.987079,0.990284,0.991386,0.991687,0.992288,0.992588,0.993089,0.992889,0.993389,0.993189,0.993389,0.990084,0.993790,0.993590,0.993089,0.993690,0.993890,0.994391,0.993690,0.994491,0.994191,0.993790,0.994191,0.994191]
acc_alt_mnist=[0.823117,0.877504,0.885717,0.890825,0.985777,0.989283,0.988381,0.991086,0.989884,0.991587,0.992087,0.992488,0.991186,0.992688,0.990885,0.991486,0.993389,0.992288,0.992889,0.992087,0.993790,0.994291,0.993389,0.994692,0.993790,0.993490,0.993590,0.992889,0.994191,0.994091]
acc_con_cifar=[0.274239,0.386819,0.459936,0.497196,0.545272,0.581530,0.624399,0.637019,0.646334,0.673377,0.632612,0.687500,0.693910,0.675280,0.712640,0.708433,0.702624,0.719351,0.711639,0.717147,0.710537,0.707332,0.727163,0.737580,0.729167,0.746394,0.730268,0.746494,0.740485,0.721655,0.746494,0.739483,0.743690,0.739183,0.736679,0.752504,0.750601,0.750501,0.744591,0.737079,0.755609,0.746194,0.756210,0.739383,0.754607,0.747496,0.757612,0.753906,0.752003,0.757111]
acc_alt_cifar=[0.264623,0.352965,0.441506,0.499099,0.530749,0.584435,0.587540,0.609876,0.644531,0.634415,0.656050,0.663261,0.688401,0.672977,0.696915,0.695913,0.709836,0.709235,0.706430,0.689303,0.715044,0.712941,0.715845,0.718450,0.722256,0.720453,0.724459,0.711238,0.730268,0.721554,0.700521,0.735276,0.738982,0.739884,0.736779,0.738482,0.738582,0.729267,0.744992,0.738281,0.742488,0.742488,0.742488,0.728866,0.742087,0.738181,0.732171,0.742588,0.739884,0.734475]
acc_alt_mnist_s=[0.873498,0.882612,0.885917,0.886819,0.886118,0.889724,0.888822,0.891026,0.889323,0.890425,0.890825,0.892228,0.890525,0.891927,0.892127,0.891827,0.892428,0.892228,0.893129,0.892929,0.892428,0.891426,0.892728,0.892829,0.892428,0.892228,0.891727,0.893129,0.892829,0.893329]

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(acc_con_mnist,color='blue',linestyle='-',linewidth=1,marker='.',label='Joint Optimization')
ax.plot(acc_alt_mnist,color='r',linestyle='-',linewidth=1,marker='.',label='Alternate Optimization')
ax.set_title('NDF Accuracy MNIST')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
spacex=1
spacey=0.005
locx = MultipleLocator(spacex)
locy = MultipleLocator(spacey)
ax.yaxis.set_minor_locator(locy)
ax.xaxis.set_minor_locator(locx)
ax.grid(which='minor')
ax.legend()
plt.show()


fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(acc_alt_mnist_s,color='green',linestyle='-',linewidth=1,marker='.',label='Bad alternate optimization')
ax.plot(acc_alt_mnist,color='r',linestyle='-',linewidth=1,marker='.',label='Good alternate optimization')
ax.set_title('NDF Accuracy MNIST')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
spacex=1
spacey=0.005
locx = MultipleLocator(spacex)
locy = MultipleLocator(spacey)
ax.yaxis.set_minor_locator(locy)
ax.xaxis.set_minor_locator(locx)
ax.grid(which='minor')
ax.legend()
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(acc_con_cifar,color='blue',linestyle='-',linewidth=1,marker='.',label='Joint Optimization')
ax.plot(acc_alt_cifar,color='r',linestyle='-',linewidth=1,marker='.',label='Alternate Optimization')
ax.set_title('NDF Accuracy CIFAR-10')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
spacex=1
spacey=0.005
locx = MultipleLocator(spacex)
locy = MultipleLocator(spacey)
ax.yaxis.set_minor_locator(locy)
ax.xaxis.set_minor_locator(locx)
ax.grid(which='minor',linewidth=0.5)
ax.legend()
plt.show()
