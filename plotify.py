from matplotlib import rcParams
import matplotlib.font_manager
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np


class Plotify:
  def __init__(self):
    # Basic configuration
    self.use_grid = True
    rcParams['font.sans-serif'] = ['Arial']
    plt.style.use('dark_background')

    # Color Constants
    self.background_color = '#1C2024'
    self.grid_color = '#444444'
    self.legend_color = '#282D33'
    self.c_cyan = '#4FB99F'
    self.c_orange = '#F2B134'
    self.c_red = '#ED553B'
    self.c_white = '#FFFFFF'

    self.plot_colors = [self.c_orange, self.c_cyan, self.c_red]

  def boxplot(self, data, labels, title, ylabel):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(self.background_color)
    ax.set_facecolor(self.background_color)

    bplot = ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        labels=labels,
        boxprops=dict(facecolor=self.c_white, color=self.c_white),
        capprops=dict(color=self.c_white),
        whiskerprops=dict(color=self.c_white),
        flierprops=dict(markeredgecolor=self.c_white),
        medianprops=dict(color=self.c_white)
    )

    for patch, color in zip(bplot['boxes'], self.plot_colors):
        patch.set_facecolor(color)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend((bplot['boxes']), labels, loc=2, facecolor=self.legend_color)

    plt.subplots_adjust(top=0.85)
    plt.grid(self.use_grid, color=self.grid_color)

    plt.show()

  def scatter_plot(
      self,
      x_list,
      y_list,
      linewidth=0.5,
      alpha=1,
      xlabel='X label',
      ylabel='Y label',
      title='Title',
      legend_labels=('Men', 'Women'),
      vectors=[],
      mean=[],
      variance=[],
      components=[]
  ):
    fig, ax = self.get_figax()

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    for i, x in enumerate(x_list):
      ax.scatter(
          x,
          y_list[i],
          linewidths=linewidth,
          alpha=alpha,
          c=self.plot_colors[i]
      )

    # vecs = plt.quiver(eigen_vectors[0][0], eigen_vectors[0][1], eigen_values[0], angles="xy", scale_units="xy", scale=1, color='r')
    # for i, v in enumerate(vectors['vector_list']):
    #   ax.arrow(
    #     vectors['origins'][i][0], 
    #     vectors['origins'][i][1], 
    #     vectors['origins'][i][0] + v[0],
    #     vectors['origins'][i][1] + v[1],
    #     head_width=0.05, head_length=0.1, fc='k', ec='k')

    for length, vector in zip(variance, components):
      v = vector * 3 * np.sqrt(length)
      self.draw_vector(mean, mean + v, ax)
    
    plt.axis('equal');

    ax.grid(self.use_grid, color=self.grid_color)
    ax.legend(legend_labels, facecolor=self.legend_color)

    plt.show()

  def histogram(
      self,
      x_list,
      ylabel='Y label',
      xlabel='X label',
      title='Title',
      labels=('Label 1', 'Label 2')
  ):
    fig, ax = self.get_figax()

    for i, x in enumerate(x_list):
      ax.hist(x, int(np.max(x) - np.min(x)), facecolor=self.plot_colors[i])

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(labels, facecolor=self.legend_color)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

  def bar(
      self,
      x_list,
      y_list,
      ylabel='Y label',
      xlabel='X label',
      title='Title',
      ymin=0,
      ymax=None,
      linewidth=0.8,
      use_x_list_as_xticks=False
  ):
    fig, ax = self.get_figax()

    ax.bar(x_list, height=y_list, width=linewidth, color=self.c_orange)
    ax.set_ylim(ymin=ymin)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    if ymax != None:
      ax.set_ylim(ymax=ymax)

    if use_x_list_as_xticks == True:
      plt.xticks(x_list)
    
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()

  def get_figax(self):
    fig, ax = plt.subplots()

    fig.patch.set_facecolor(self.background_color)

    ax.set_facecolor(self.background_color)
    ax.grid(self.use_grid, color=self.grid_color)

    return fig, ax

  def draw_vector(self, v0, v1, ax):
    arrowprops = dict(
      arrowstyle='->',
      linewidth=1,
      shrinkA=0,
      shrinkB=0
    )

    ax.annotate('', v1, v0, arrowprops=arrowprops)
