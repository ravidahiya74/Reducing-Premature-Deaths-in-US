# Import libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Function to highlight multicolliniarity among variables
def print_corr(df, pct=0):
    sns.set(style='white')
    # Compute the correlation matrix
    if pct == 0:
        corr = df.corr()
    else:
        corr = abs(df.corr()) > pct
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5})


# Function to remove variables with high p-values and re-assigning to X_train and X_test
def remove_p(model,X_train,X_test):
    imp_features=list(model.pvalues[model.pvalues<0.05].index)[1:]
    X_train=X_train[imp_features]
    X_test=X_test[imp_features]
    return X_train, X_test

