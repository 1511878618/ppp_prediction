import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted, venn3, venn3_circles

import os.path
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse


def plot_venn2(
    Group1,
    Group2,
    ax=None,
    labels=("Group1", "Group2"),
    colors=("#0073C2FF", "#EFC000FF"),
    title=None,
    alpha=0.8,
):
    if ax is None:
        fig, ax = plt.subplots()

    if title is None:
        title = f"Venn Diagram of {labels[0]} and {labels[1]}"
    vee2 = venn2(
        [set(Group1), set(Group2)],
        # set_labels=("Group1", "Group2"),
        set_labels=labels,
        set_colors=colors,
        # set_colors=("#0073C2FF", "#EFC000FF"),
        alpha=alpha,
        ax=ax,
    )
    venn2_circles(
        [set(Group1), set(Group2)], linestyle="--", linewidth=2, color="black", ax=ax
    )

    # 定制化设置:设置字体属性
    for text in vee2.set_labels:
        text.set_fontsize(15)
    for text in vee2.subset_labels:
        try:
            text.set_fontsize(16)
            text.set_fontweight("bold")
        except:
            pass
    # ax.text(
    #     0.8,
    #     -0.1,
    #     "\nVisualization by DataCharm",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="center",
    #     fontsize=8,
    #     color="black",
    # )
    ax.set_title(
        title,
        size=15,
        fontweight="bold",
        # backgroundcolor="#868686FF",
        color="black",
        style="italic",
    )
    return ax


def plot_venn3(
    Group1,
    Group2,
    Group3,
    ax=None,
    labels=("Group1", "Group2", "Group3"),
    colors=("#0073C2FF", "#EFC000FF", "#CD534CFF"),
    title=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    if title is None:
        title = f"Venn Diagram of {labels[0]}, {labels[1]} and {labels[2]}"
    vd3 = venn3(
        [set(Group1), set(Group2), set(Group3)],
        set_labels=labels,
        set_colors=colors,
        alpha=0.8,
        ax=ax,
    )
    venn3_circles(
        [set(Group1), set(Group2), set(Group3)],
        linestyle="--",
        linewidth=2,
        color="black",
        ax=ax,
    )
    for text in vd3.set_labels:
        text.set_fontsize(15)
        text.set_fontweight("bold")
    for text in vd3.subset_labels:
        try:
            text.set_fontsize(15)
            text.set_fontweight("bold")
        except:
            pass
    # ax.text(
    #     0.8,
    #     -0.1,
    #     "\nVisualization by DataCharm",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="center",
    #     fontsize=9,
    #     color="black",
    # )
    ax.set_title(
        title,
        fontweight="bold",
        fontsize=15,
        pad=30,
        # backgroundcolor="#868686FF",
        color="black",
        style="italic",
    )
    return ax




#get shared elements for each combination of sets
def get_shared(sets):
    IDs = sets.keys()
    combs = sum([list(map(list, combinations(IDs, i))) for i in range(1, len(IDs) + 1)], [])

    shared = {}
    for comb in combs:
        ID = ' and '.join(comb)
        if len(comb) == 1:
            shared.update({ID: sets[comb[0]]})
        else:
            setlist = [sets[c] for c in comb]
            u = set.intersection(*setlist)
            shared.update({ID: u})
    return shared


#get unique elements for each combination of sets
def get_unique(shared):
    unique = {}
    for shar in shared:
        if shar == list(shared.keys())[-1]:
            s = shared[shar]
            unique.update({shar: s})
            continue
        count = shar.count(' and ')
        if count == 0:
            setlist = [shared[k] for k in shared.keys() if k != shar and " and " not in k]
            s = shared[shar].difference(*setlist)
        else:
            setlist = [shared[k] for k in shared.keys() if k != shar and k.count(' and ') >= count]
            s = shared[shar].difference(*setlist)
        unique.update({shar: s})
    return(unique)


#plot Venn
def venny4py(
        sets=(),
        out='./',
        asax=False,
        ext='png',
        dpi=300,
        size=3.5,
        colors="bgrc",
        line_width=None,
        font_size=None,
        legend_cols=2,
        column_spacing=4,
        edge_color='black'
):
    if len(sets) < 2 or len(sets) > 4:
        raise ValueError('Number of sets must be 2, 3 or 4')
    shared = get_shared(sets)
    unique = get_unique(shared)
    ce = colors
    lw = size*.12 if line_width is None else line_width
    fs = size*2 if font_size is None else font_size
    nc = legend_cols
    cs = column_spacing
    ec = edge_color

    os.makedirs(out, exist_ok=True)
    
    with open(f'{out}/Intersections_{len(sets)}.txt', 'w') as f:
        for k, v in unique.items():
            f.write(f'{k}: {len(v)}, {sorted(list(v))}\n')
    
    if asax == False:
        plt.rcParams['figure.dpi'] = 200 #dpi in notebook
        plt.rcParams['savefig.dpi'] = dpi #dpi in saved figure
        fig, ax = plt.subplots(1, 1, figsize=(size, size))

    else:
        ax = asax
        
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    #4 sets
    if len(sets) == 4:
        #draw ellipses
        ew = 45 #width
        eh = 75 #height
        xe = [35, 48, 52, 65] #x coordinats
        ye = [35, 45, 45, 35] #y coordinats
        ae = [225, 225, 315, 315] #angles

        for i, s in enumerate(sets):
            ax.add_artist(Ellipse(xy=(xe[i], ye[i]), width=ew, height=eh, fc=ce[i], 
                                  angle=ae[i], alpha=.3))
            ax.add_artist(Ellipse(xy=(xe[i], ye[i]), width=ew, height=eh, fc='None',
                                  angle=ae[i], ec=ec, lw=lw))

        #annotate
        xt = [12, 32, 68, 88, 14, 34, 66, 86, 26, 28, 50, 50, 72, 74, 37, 60, 40, 63, 50] #x
        yt = [67, 79, 79, 67, 41, 70, 70, 41, 59, 26, 11, 60, 26, 59, 51, 17, 17, 51, 35] #y

        for j, s in enumerate(sets):
            ax.text(xt[j], yt[j], len(sets[s]), ha='center', va='center', fontsize=fs, 
                    transform=ax.transData)

        for k in unique:
            j += 1
            ax.text(xt[j], yt[j], len(unique[k]), ha='center', va='center', fontsize=fs, 
                    transform=ax.transData)
            
    #3 sets
    if len(sets) == 3:
        #draw circles
        ew = 60 #width
        eh = 60 #height
        lw = size*.12 #line width
        xe = [37, 63, 50] #x coordinats
        ye = [55, 55, 32] #y coordinats
        nc = 3 #legend columns
        cs = 1 #columns spacing

        for i, s in enumerate(sets):
            ax.add_artist(Ellipse(xy=(xe[i], ye[i]), width=ew, height=eh, fc=ce[i], 
                                  angle=0, alpha=.3))
            ax.add_artist(Ellipse(xy=(xe[i], ye[i]), width=ew, height=eh, fc='None',
                                  angle=0, ec=ec, lw=lw))

        #annotate
        xt = [12, 88, 28, 22, 78, 50, 50, 30, 70, 50] #x
        yt = [80, 80,  3, 60, 60, 17, 70, 35, 35, 50] #y
        

        for j, s in enumerate(sets):
            ax.text(xt[j], yt[j], len(sets[s]), ha='center', va='center', fontsize=fs, 
                    transform=ax.transData)

        for k in unique:
            j += 1
            ax.text(xt[j], yt[j], len(unique[k]), ha='center', va='center', fontsize=fs, 
                    transform=ax.transData)
            
    #2 sets
    if len(sets) == 2:
        #draw circles
        ew = 70 #width
        eh = 70 #height
        lw = size*.12 #line width
        xe = [37, 63] #x coordinats
        ye = [45, 45] #y coordinats

        for i, s in enumerate(sets):
            ax.add_artist(Ellipse(xy=(xe[i], ye[i]), width=ew, height=eh, fc=ce[i], 
                                  angle=0, alpha=.3))
            ax.add_artist(Ellipse(xy=(xe[i], ye[i]), width=ew, height=eh, fc='None',
                                  angle=0, ec=ec, lw=lw))

        #annotate
        xt = [20, 80, 18, 82, 50] #x
        yt = [80, 80, 45, 45, 45] #y

        for j, s in enumerate(sets):
            ax.text(xt[j], yt[j], len(sets[s]), ha='center', va='center', fontsize=fs, 
                    transform=ax.transData)

        for k in unique:
            j += 1
            ax.text(xt[j], yt[j], len(unique[k]), ha='center', va='center', fontsize=fs, 
                    transform=ax.transData)
                
    #legend
    handles = [mpatches.Patch(color=ce[i], label=l, alpha=.3) for i, l in enumerate(sets)]
    ax.legend(labels=sets, handles=handles, fontsize=fs*1.1, frameon=False, 
              bbox_to_anchor=(.5, .99), bbox_transform=ax.transAxes, loc=9, 
              handlelength=1.5, ncol=nc, columnspacing=cs, handletextpad=.5)
    if asax == False:
        fig.savefig(f'{out}/Venn_{len(sets)}.{ext}', bbox_inches='tight', facecolor='w', )