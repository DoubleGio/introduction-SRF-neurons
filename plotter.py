import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from typing import Literal

# Plotting functions

def snn_plot(
    mem: list|np.ndarray = None,
    spk_in: list|np.ndarray = None,
    cur_in: list|np.ndarray = None,
    spk_out: list|np.ndarray = None,
    title: str = None,
    ylim: float | list | np.ndarray = None,
    vline: int|list|np.ndarray = None,
    hline: float|list|np.ndarray = None,
    hline_label: str|list[str] = "Threshold",
    dt: float = None,
    return_fig: bool = False,
    mem_labels: str|list[str] = None,
    fig_size: tuple = (10, 3.8),  # Default figure size is (10, 4.8) inches
    phase_portrait: bool = False
) -> None|plt.Figure:
    """
    Plot input spikes, input current, membrane potential, and output spikes.
    
    Parameters:
        mem: Membrane potential.
        spk_in: Input spikes.
        cur_in: Input current.
        spk_out: Output spikes.
        title: Title of the plot.
        ylim: Y-axis limits. For separate limits per trace, provide a 2D array with shape (n_traces, ...).
        vline: Vertical line position(s).
        hline: Horizontal line to plot over membrane potential.
        hline_label: Label for the horizontal line(s).
        dt: Time step for x-axis label.
        return_fig: If True, return the figure object instead of showing it.
        mem_labels: Labels for the membrane potential traces.
        fig_size: Size of the figure in inches (width, height).
        phase_portrait: If True, add phase portrait plots to the right of membrane plots.
    Returns:
        None or plt.Figure if return_fig is True.
    """
    if spk_in is not None and cur_in is not None:
        raise ValueError("Only one of `spk_in` or `cur_in` may be provided, not both.")
    c = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 2
    spk_in = _arr_check(spk_in)
    cur_in = _arr_check(cur_in)
    mem = _arr_check(mem)
    spk_out = _arr_check(spk_out)

    plot_grid = [
        *( [min(1, max(0.1 * spk_in.shape[0], 0.4)) if spk_in.shape[0] > 1 else 0.4] * bool(spk_in.shape[0] or cur_in.shape[0]) ),
        *( [1] * mem.shape[0] ),
        *( [min(1, max(0.1 * spk_out.shape[0], 0.4))] * bool(spk_out.shape[0]) )
    ]
    
    # Calculate figure layout based on phase portrait option
    fig_size = (fig_size[0], fig_size[1] + sum(plot_grid))
    if phase_portrait and mem.size > 0:
        fig, ax_ = plt.subplots(
            len(plot_grid), 2, figsize=fig_size, sharex='col', sharey='row', squeeze=False, layout='constrained',
            gridspec_kw = {'width_ratios': [3.0, 1.0], 'height_ratios': plot_grid},
        )
        ax_[0, 1].remove()
        ax_[-1, 1].remove()
        ax = ax_[:, 0]
        ax_phase = ax_[1:-1, 1]
    else:
        fig, ax = plt.subplots(
            len(plot_grid), figsize=fig_size, sharex=True, squeeze=False, layout='constrained',
            gridspec_kw = {'height_ratios': plot_grid},
        )
        ax = ax.flatten()

    ax_count = 0
    if title:
        ax[ax_count].set_title(title)

    # Plot input spikes
    if spk_in.size > 0:
        _plot_spk(spk_in, ax[ax_count], c, output=False)
        ax_count += 1

    # Plot input current
    elif cur_in.size > 0:
        if cur_in.shape[0] > 1:
            for i in range(cur_in.shape[0]):
                ax[ax_count].plot(cur_in[i], c=c[i], label=f"Input {i+1}", alpha=0.75)
            ax[ax_count].legend(loc='upper right', ncol=int(np.ceil(cur_in.shape[0]/2)), handlelength=1.5, handletextpad=0.5, columnspacing=1., fontsize='small')
        else:
            ax[ax_count].plot(cur_in[0], c="tab:green", label="Input")
        ax[ax_count].set(ylabel="Input\nCurrent", xlim=[0, len(cur_in[0])-1])
        ax_count += 1

    # Plot membrane potential
    if mem.size > 0:
        if mem.shape[0] > 1:
            for i in range(mem.shape[0]):
                ax[ax_count].plot(mem[i].real, c=c[i], label=f"U.real {i+1}")
                ax[ax_count].plot(mem[i].imag, c=c[i], alpha=0.8, linestyle="--", label=f"U.imag {i+1}")
                _plot_hline(hline, hline_label, ax[ax_count], trace_idx=i)
                ax[ax_count].set(ylabel=f"$U_{{\\rm mem {i+1}}}$", xlim=[0, len(mem[i])-1])
                ax[ax_count].legend(loc='upper right', fontsize='small', handlelength=1.5)
                _set_ylim(ylim, ax[ax_count], i)
                _label_mem(mem_labels, ax[ax_count], trace_idx=i)
                if phase_portrait:
                    _plot_single_phase_portrait(mem, spk_out, ax_phase[i], c[i], hline, ylim, i)
                ax_count += 1
        else:
            ax[ax_count].plot(mem[0].real, c="tab:blue", label="Real")
            ax[ax_count].plot(mem[0].imag, c="tab:orange", linestyle="--", label="Imaginary")
            _plot_hline(hline, hline_label, ax[ax_count])
            ax[ax_count].set(ylabel=f"$U_{{\\rm mem}}$", xlim=[0, len(mem[0])-1])
            ax[ax_count].legend(loc='upper right', fontsize='small', handlelength=1.5)
            _set_ylim(ylim, ax[ax_count])
            _label_mem(mem_labels, ax[ax_count])            
            if phase_portrait:
                _plot_single_phase_portrait(mem, spk_out, ax_phase[0], "tab:blue", hline, ylim, 0)
            ax_count += 1

    # Plot output spikes
    if spk_out.size > 0:
        _plot_spk(spk_out, ax[ax_count], c, output=True)
        ax_count += 1

    # Set xlabel only on the bottom time domain plot
    if len(ax) > 0:
        ax[-1].set_xlabel(f"Time steps \u0394t={dt}" if dt else "Time steps")
    
    if vline is not None:
        if isinstance(vline, (list, np.ndarray)):
            for a in ax:
                if isinstance(vline, (list, np.ndarray)):
                    for v in vline:
                        a.axvline(v, color="black", linestyle="--", alpha=0.25, linewidth=1)
                else:
                    a.axvline(vline, color="black", linestyle="--", alpha=0.25, linewidth=1)

    # Add labels to phase portraits
    if phase_portrait:
        for i, phase_ax in enumerate(ax_phase):
            phase_ax.set_xlabel("real", fontsize='small', alpha=0.75)
            phase_ax.xaxis.set_label_coords(0.5, 0.075)
            phase_ax.set_ylabel("imag", fontsize='small', alpha=0.75, rotation=90)
            phase_ax.yaxis.set_label_coords(0.075, 0.5)
        ax_phase[-1].xaxis.set_tick_params(labelbottom=True)

    if return_fig:
        return fig
    else:
        plt.show()

def _set_ylim(ylim, ax, trace_idx=None):
    """Set the y-axis limits for a plot."""
    if ylim is None:
        ax.set_ylim([-1, 1])
    elif isinstance(ylim, np.ndarray):
        if ylim.ndim == 2 and trace_idx is not None:
            if ylim.shape[1] == 2:
                ax.set_ylim([ylim[trace_idx, 0], ylim[trace_idx, 1]])
            else:
                ax.set_ylim([-ylim[trace_idx], ylim[trace_idx]])
        elif ylim.ndim == 1:
            if trace_idx is not None:
                ax.set_ylim([-ylim[trace_idx], ylim[trace_idx]])
            elif len(ylim) == 2:
                ax.set_ylim([ylim[0], ylim[1]])
            else:
                ax.set_ylim([-ylim[0], ylim[0]])
        else:
            raise ValueError(f"Unsupported `ylim` np.ndarray shape: {ylim.shape}")
    elif isinstance(ylim, list):
        if len(ylim) != 2:
            raise ValueError(f"`ylim` list must have len 2, got {len(ylim)}")
        ax.set_ylim([ylim[0], ylim[1]])
    elif isinstance(ylim, int) or isinstance(ylim, float):
        ax.set_ylim([-ylim, ylim])

def _arr_check(arr: list|np.ndarray) -> np.ndarray:
    """Convert a list or numpy array to a 2D numpy array (n arrays x m samples)"""
    if arr is None:
        return np.array([])
    else:
        arr = np.array(arr)
        if arr.ndim > 2:
            raise ValueError(f"Input array has more than 2 dimensions: {arr.ndim}")
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr

def _plot_hline(hline: float|list|np.ndarray, hline_label: str|list[str], ax, trace_idx = None):
    if hline is None:
        return
    hline = np.array(hline)
    if hline.ndim == 0: # Single value hline
        ax.axhline(y=hline.item(), color="black", linestyle=":", alpha=0.4, linewidth=1.5, label=hline_label)
    elif hline.ndim == 1:
        if trace_idx is not None: # Single value per trace
            ax.axhline(y=hline[trace_idx], color="black", linestyle=":", alpha=0.4, linewidth=1.5, 
                    label=hline_label if isinstance(hline_label, str) else hline_label[trace_idx])
        else: # Time-varying shared threshold
            ax.plot(hline, color="black", linestyle=":", alpha=0.4, linewidth=1.5, label=hline_label)
    elif hline.ndim == 2 and trace_idx is not None:
        ax.plot(hline[trace_idx], color="black", linestyle=":", alpha=0.4, linewidth=1.5, 
                label=hline_label if isinstance(hline_label, str) else hline_label[trace_idx])
    else:
        raise ValueError(f"Unsupported `thr_line` shape: {hline.shape}")

def _label_mem(mem_labels: str|list[str], ax, trace_idx: int = None):
    if mem_labels is None:
        return
    if isinstance(mem_labels, str):
        ax.text(0.01, 0.01, mem_labels, transform=ax.transAxes, fontsize='small', va='bottom', ha='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    elif isinstance(mem_labels, list):
        if trace_idx is None:
            raise ValueError("trace_idx must be provided when mem_labels is a list.")
        if trace_idx < len(mem_labels):
            ax.text(0.01, 0.01, mem_labels[trace_idx], transform=ax.transAxes, fontsize='small', va='bottom', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            # alternatively use annotate
            # ax.annotate(mem_labels[trace_idx], xy=(0, 0), xycoords='axes fraction', xytext=(+0.5, +0.5),
            # textcoords='offset fontsize', fontsize='medium', verticalalignment='bottom', 
            # bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.9, edgecolor="none"))
        else:
            raise ValueError(f"trace_idx {trace_idx} exceeds length of mem_labels {len(mem_labels)}")

def _plot_spk(spk_arr, ax, c, output=False):
    w = max(1, 2-np.log10(spk_arr.shape[-1]/10))
    s = 400 / spk_arr.shape[0]
    pos_y, pos_x = np.where(spk_arr > 0)
    neg_y, neg_x = np.where(spk_arr < 0)

    if spk_arr.shape[0] == 1: # Simple spk_plot
        ax.vlines(pos_x, ymin=-0.4, ymax=0.4, linewidth=w, colors='black', label="Pos.")
        if neg_x.size > 0:
            ax.vlines(neg_x, ymin=-0.4, ymax=0.4, linewidth=w, colors='red', label="Neg.")
            ax.legend(loc='upper right', handlelength=1.5, handletextpad=0.5, fontsize='small')
        ax.set(ylabel=f"{'Output' if output else 'Input'}\nSpikes", xlim=[0, spk_arr.shape[-1]], yticks=[], ylim=[-1, 1])
    else: # Scatter spk_plot
        for i in range(spk_arr.shape[0]):
            pos_x_i, pos_y_i = pos_x[pos_y == i], pos_y[pos_y == i]
            if pos_x_i.size > 0:
                ax.scatter(pos_x_i, pos_y_i, c=c[i], marker='|', s=s, linewidth=w)
            neg_x_i, neg_y_i = neg_x[neg_y == i], neg_y[neg_y == i]
            if neg_x_i.size > 0:
                ax.scatter(neg_x_i, neg_y_i, c='red', marker='|', s=s, linewidth=w)

        yticks = np.arange(0, spk_arr.shape[0], np.ceil(spk_arr.shape[0] / 8), dtype=int)[::-1]
        yticklabels = yticks + 1
        ax.set(ylabel=f"{'Output' if output else 'Input'}\nSpikes", yticks=yticks, yticklabels=yticklabels.astype(str),
                xlim=[0, spk_arr.shape[1]], ylim=[-0.5, spk_arr.shape[0]-1+0.5])
        if output:
            ax.invert_yaxis()
        if spk_arr.shape[0] > 8:
            yticks_minor = np.setdiff1d(np.arange(spk_arr.shape[0]), yticks)
            ax.set_yticks(yticks_minor, minor=True)
            ax.grid(which='major', axis='y', color='gray', linestyle='--', linewidth=0.75, alpha=0.5)

### PHASE PORTRAIT PLOT FUNCTIONS ###
def _plot_single_phase_portrait(
        mem, 
        spk_out, 
        ax, 
        color, 
        thrline, 
        xylim, 
        trace_idx,
        thrplane="imag",
        label_lines=False
    ):
    """
    Plot a single phase portrait.
    """
    if xylim is None:
        xylim = [-1.0, 1.0]
    elif isinstance(xylim, (int, float)):
        xylim = [-xylim, xylim]
    x, y = mem[trace_idx].real, mem[trace_idx].imag
    trace, = ax.plot(x, y, c=color, alpha=0.8, label=(f"$U_{{\\rm mem {trace_idx+1}}}$" if mem.shape[0] > 1 else "$U_{\\rm mem}$"))
    start, = ax.plot(x[0], y[0], 'o', c=color, markersize=5, markeredgecolor="black", markeredgewidth=0.75, label="Start")
    # ax.plot(x[-1], y[-1], 's', c=color, markersize=4, markeredgecolor="black", markeredgewidth=0.75, label="End")
    
    # Plot spikes on phase portrait if available
    if spk_out.size > 0:
        spk_idx = np.where(spk_out[trace_idx])[0]
        if len(spk_idx) > 0:
            spike, = ax.plot(x[spk_idx-1], y[spk_idx-1], '*', c="red", markeredgecolor='black', markeredgewidth=0.75, markersize=5, label="Spike", alpha=0.8)

    thrline = np.array(thrline) if thrline is not None else np.array([])
    if thrline.size > 0:
        if thrline.ndim == 0:
            thr_val = thrline.item()
        elif thrline.ndim == 1:
            if thrline.shape[0] == mem.shape[1]: # Shared time-varying threshold
                raise ValueError("Can only print time-varying threshold if `animate` is True.")
            elif thrline.shape[0] == mem.shape[0]: # Single value per trace
                thr_val = thrline[trace_idx]
        elif thrline.ndim == 2: # Time-varying threshold for each trace
            raise ValueError("Can only print time-varying threshold if `animate` is True.")
        
        if thrplane == "real":
            thr = ax.axvline(x=thr_val, color="black", linestyle=":", alpha=0.4, linewidth=1.5, label="Thr.")
        elif thrplane == "imag":
            thr = ax.axhline(y=thr_val, color="black", linestyle=":", alpha=0.4, linewidth=1.5, label="Thr.")
    if label_lines:
        line_legend = ax.legend(handles=[trace, thr], loc='upper left', fontsize='small', handlelength=1., handletextpad=0.5, framealpha=0.8)
        ax.add_artist(line_legend)
    ax.legend(handles=[start, spike], loc='lower right', fontsize='small', handlelength=1., handletextpad=0.5, framealpha=0.8)
    ax.set(xlim=[xylim[0], xylim[1]], ylim=[xylim[0], xylim[1]], aspect='equal', adjustable='box')

def plot_phase_portrait(
    mem: list|np.ndarray = None,
    spk: list|np.ndarray = None,
    title: str = None,
    xylim: float = 1,
    thrline: float|list|np.ndarray = None,
    thrplane: Literal["real", "imag"] = "imag",
    return_fig: bool = False,
    mem_labels: str|list[str] = None,
    animate: bool = False,
    dt: float = 0.01,
    animate_speed: float = 1.0,
    max_fps: int = 60
) -> None|plt.Figure:
    """
    Plot phase portraits of membrane potentials with optional spike markers and threshold lines.
    Parameters:
        mem: Membrane potentials as a list or numpy array of shape (n_traces, n_samples).
        spk: Spikes as a list or numpy array of shape (n_traces, n_samples).
        title: Title of the plot.
        xylim: Limits for the x and y axes.
        thrline: Threshold line(s) to plot, can be a single value, list, or numpy array.
        thrplane: Plane on which to plot the threshold line ("real" or "imag").
        return_fig: If True, return the figure object instead of showing it.
        mem_labels: Labels for each membrane potential trace.
        animate: If True, create an animated phase portrait.
        dt: Time step for animation frames.
        animate_speed: Speed multiplier for the animation.
    """
    def update(n):
        for i, (trace, arrow, spike, line) in enumerate(zip(traces, arrows, spikes, lines)):
            trace.set_data(mem[i].real[:n+1], mem[i].imag[:n+1])
            if n > 0:
                dx = mem[i].real[n] - mem[i].real[n-1]
                dy = mem[i].imag[n] - mem[i].imag[n-1]
                if np.sqrt(dx**2 + dy**2) > 0:
                    arrow.set_alpha(1.0)
                    arrow.set_positions((mem[i].real[n], mem[i].imag[n]), (mem[i].real[n] + dx*0.1, mem[i].imag[n] + dy*0.1))
                else:
                    arrow.set_positions((mem[i].real[n], mem[i].imag[n]), (mem[i].real[n] + 0.001, mem[i].imag[n]))
            if spk.size > 0:
                spk_idx = np.where(spk[i][:n+1])[0]
                spike.set_data(mem[i].real[spk_idx-1], mem[i].imag[spk_idx-1])
            if thrline.size > 0:
                if thrline.ndim == 0:
                    thr_val = thrline.item()
                elif thrline.ndim == 1:
                    if thrline.shape[0] == mem.shape[1]:  # Shared time-varying threshold
                        thr_val = thrline[min(n, len(thrline)-1)]
                    elif thrline.shape[0] == mem.shape[0]:  # Single value per trace
                        thr_val = thrline[i]
                elif thrline.ndim == 2:
                    thr_val = thrline[i, min(n, thrline.shape[1]-1)]
                
                if thrplane == "real":
                    line.set_data([thr_val, thr_val], [-xylim, xylim])
                elif thrplane == "imag":
                    line.set_data([-xylim, xylim], [thr_val, thr_val])
        return traces + arrows + spikes + lines

    mem = _arr_check(mem)
    spk = _arr_check(spk)
    if spk.size != 0 and mem.size != spk.size:
        raise ValueError("`mem` and `spk` must have the same number of traces.")
    thrline = np.array(thrline) if thrline is not None else np.array([])

    # Determine grid shape automatically based on number of traces
    n_traces = mem.shape[0]
    if n_traces == 0:
        raise ValueError("No traces provided for phase portrait plot.")
    if n_traces < 3:
        nrows, ncols = 1, n_traces
        fs = (4, 4) if n_traces == 1 else (7, 4)
    elif np.sqrt(n_traces) % 1 == 0:
        ncols = nrows = int(np.sqrt(n_traces))
        fs = (2 + 2 * ncols, 2 + 2 * nrows)
    else: # in rows of 3
        ncols = 3
        nrows = int(np.ceil(n_traces / ncols))
        fs = (2 + 2 * ncols, 1 + 2 * nrows)
    fig, ax = plt.subplots(nrows, ncols, figsize=fs, layout='constrained', sharex=True, sharey=True, squeeze=False)
    ax = ax.flatten()
    
    if animate:
        traces, arrows, spikes, lines = [], [], [], []
        for i in range(n_traces):
            trace, = ax[i].plot([], [], c="tab:blue", label=f"$U_{{mem {i+1}}}$")
            traces.append(trace)
            arrow = FancyArrowPatch((mem[i].real[0], mem[i].imag[0]), (mem[i].real[0], mem[i].imag[0]), 
                                    arrowstyle='->', mutation_scale=15, color="tab:blue", linewidth=2, alpha=0.0)
            ax[i].add_patch(arrow)
            arrows.append(arrow)
            start, = ax[i].plot([mem[i].real[0]], [mem[i].imag[0]], 'o', c="tab:blue", markersize=5, markeredgecolor="black", markeredgewidth=0.75, label="Start")
            spike, = ax[i].plot([], [], '*', c="red", markeredgecolor='black', markeredgewidth=0.75, markersize=5, label="Spike")
            spikes.append(spike)
            line, = ax[i].plot([], [], c="black", linestyle=":", alpha=0.4, linewidth=1.5, label="Thr.")
            lines.append(line)
            line_legend = ax[i].legend(handles=[trace, line], loc='upper left', fontsize='small', handlelength=1., handletextpad=0.5, framealpha=0.8)
            ax[i].add_artist(line_legend)
            ax[i].legend(handles=[start, spike], loc='lower right', fontsize='small', handlelength=1., handletextpad=0.5, framealpha=0.8)
        # Calculate frame skipping to maintain timing with max_fps constraint
        frame_interval = 1000 * dt / animate_speed  # milliseconds per frame
        max_interval = 1000 / max_fps
        skip_factor = int(np.ceil(max_interval / frame_interval)) if frame_interval < max_interval else 1

        ani = FuncAnimation(
            fig, update, frames=range(0, len(mem[i]), skip_factor),
            blit=True, repeat=False, interval=frame_interval * skip_factor,
        )
    else:
        for i in range(n_traces):
            _plot_single_phase_portrait(
                mem, spk, ax[i], "tab:blue", thrline, xylim, i, thrplane=thrplane, label_lines=True)

    if title:
        fig.suptitle(title)
    fig.supxlabel("real", fontsize='medium')
    fig.supylabel("imag", fontsize='medium')
    
    for i, a in enumerate(ax):
        a.set(xlim=[-xylim, xylim], ylim=[-xylim, xylim], 
              aspect='equal', adjustable='box')
        if xylim <= 5:  # For small ranges, use all integers
            a.set_xticks(np.arange(-int(xylim), int(xylim)+1))
            a.set_yticks(np.arange(-int(xylim), int(xylim)+1))
        else:  # For larger ranges, use fewer ticks
            step = max(1, int(xylim/5))
            a.set_xticks(np.arange(-int(xylim), int(xylim)+1, step))
            a.set_yticks(np.arange(-int(xylim), int(xylim)+1, step))
            a.tick_params(axis='both', which='major', labelsize=8)
        if mem_labels:
            _label_mem([' ' + l for l in mem_labels] if isinstance(mem_labels, list) else ' ' + mem_labels, a, i)
    for i in range(n_traces, nrows * ncols):
        ax[i].set_visible(False)
    
    if animate:
        plt.close()
        return ani
    elif return_fig:
        return fig
    else:
        plt.show()
