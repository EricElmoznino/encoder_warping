import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # layer 3
    # Read data
    plt.clf()
<<<<<<< HEAD
    df3 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=nsd_arch=resnet18_dataset=imagenet_task=object-classification_layer=avgpool/results.csv')
    df4 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=nsd_arch=resnet18_dataset=None_task=object-classification_layer=avgpool/results.csv')
=======
    df3 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=nsd_arch=resnet18_dataset=imagenet_task=object-classification_layer=layer4/results.csv')
    df4 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=nsd_arch=resnet18_dataset=None_task=object-classification_layer=layer4/results.csv')
>>>>>>> parent of e90e11a... added plots for majaj and nsd
    # gca stands for 'get current axis'
    df3['low_dim'] = np.log10(df3['low_dim'])
    df4['low_dim'] = np.log10(df4['low_dim'])
    ax = plt.gca()

    df3.plot(kind='line',x='low_dim',y='test_r2',ax=ax, label='Res18Pre')
    df4.plot(kind='line',x='low_dim',y='test_r2', color='red', ax=ax, label='Res18NoPre')
    plt.xlabel('Number of dimensions')
    plt.ylabel('R^2')
<<<<<<< HEAD
    plt.title('Resnet18 avgpool Nsd')
    plt.savefig('/home/swadhwa5/projects/encoder_warping/simulations/dim_complexity/saved_figures/nsd/dim_R2_plot_avgpool.png')
=======
    plt.title('Resnet18 layer3')
    plt.savefig('/home/swadhwa5/projects/encoder_warping/simulations/dim_complexity/saved_figures/dim_R2_plot_layer4.png')

>>>>>>> parent of e90e11a... added plots for majaj and nsd

if __name__ == "__main__":
    main()