function make_figures_r0()
% Reproduce the DC-microgrid case-study figures.
% Three full-width case panels, Times New Roman ticks/legends, dash-dot curves,
% original green/purple/blue/red palette, boxed legends, and Helvetica insets.
root = fileparts(mfilename('fullpath'));
output = fullfile(root, 'figures', 'r0_style');
if ~exist(output, 'dir'), mkdir(output); end
d = load(fullfile(root, 'data', 'three_case_results.mat'));
cases = {d.case1, d.case2, d.case3};
colors = [0.2080 0.7188 0.4670; 0.2429 0.1689 0.4270; ...
          0.0000 0.4509 0.7402; 1.0000 0.0000 0.0000];
specs = {
 'Fig_A_1_voltage', 'voltage_deviation', '\Delta V', true, false;
 'Fig_A_2_sharing', 'sharing_error', 'e_I', true, false;
 'Fig_A_3_current', 'I_t', 'I_t', false, true;
 'Fig_A_4_control', 'u', 'u', true, false;
 'Fig_A_5_observer', 'observer_error', '\widetilde I_L', true, false;
 'Fig_A_6_lines', 'I_line', 'I', false, true;
 'Fig_A_7_loads', 'load', 'I_L', false, true;
};
for k = 1:size(specs, 1)
    fig = figure('Visible', 'off', 'Color', 'white', 'Units', 'points', ...
                 'Position', [50 50 560 560], 'Renderer', 'painters');
    for ci = 1:3
        c = cases{ci}; t = c.t(:); y = c.(specs{k, 2});
        if ci == 2 && ismember(specs{k,2}, {'load','sharing_error','observer_error'})
            % Plot both one-sided values at a load jump. Dynamic states stay
            % continuous; only the actual load and load-based errors jump.
            for event = [8 6 4 2]
                index = find(t == event, 1);
                if strcmp(specs{k,2}, 'sharing_error')
                    left = y(index,:) + c.lambda_d(index) - c.lambda_d(index-1);
                else
                    left = y(index,:) - (c.load(index,:) - c.load(index-1,:));
                end
                t = [t(1:index-1); event; t(index:end)];
                y = [y(1:index-1,:); left; y(index:end,:)];
            end
        end
        bottom = 0.755 - (ci - 1) * 0.325;
        pos = [0.100 bottom 0.875 0.220];
        ax = axes(fig, 'Position', pos); hold(ax, 'on');
        handles = gobjects(1, 4); names = cell(1, 4);
        for i = 1:4
            handles(i) = plot(ax, t, y(:,i), '-.', 'Color', colors(i,:), 'LineWidth', 1.5);
            if strcmp(specs{k, 2}, 'I_line')
                suffixes = {'12', '23', '34', '14'};
                names{i} = sprintf('$I_{%s}$, Case %d', suffixes{i}, ci);
            elseif strcmp(specs{k, 2}, 'I_t')
                names{i} = sprintf('$I_{t%d}$, Case %d', i, ci);
            elseif strcmp(specs{k, 2}, 'sharing_error')
                names{i} = sprintf('$e_{I%d}$, Case %d', i, ci);
            elseif strcmp(specs{k, 2}, 'observer_error')
                names{i} = sprintf('$%s I_{L%d}$, Case %d', '\widetilde', i, ci);
            elseif strcmp(specs{k, 2}, 'load')
                names{i} = sprintf('$I_{L%d}$, Case %d', i, ci);
            else
                names{i} = sprintf('$%s_{%d}$, Case %d', specs{k,3}, i, ci);
            end
        end
        if strcmp(specs{k, 2}, 'voltage_deviation')
            handles(end+1) = plot(ax, t, c.voltage_error(:), 'k--', 'LineWidth', 2);
            names{end+1} = '$\mathcal{R}_V$';
        end
        set(ax, 'FontName', 'Times New Roman', 'FontSize', 18, ...
            'LineWidth', 1, 'Box', 'on', 'TickDir', 'in', 'Layer', 'top', ...
            'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], ...
            'XLim', [0 10], 'XTick', 0:2:10, 'XGrid', 'off', 'YGrid', 'off');
        xlabel(ax, 'Time (s)', 'FontName', 'Times New Roman', 'FontSize', 19.8);
        ax.XAxis.Exponent = 0;
        % Machine-flat panels (Case-1 residuals) otherwise auto-scale to junk
        % decades and print a floating x10^n label clipped at the canvas edge.
        ax.YAxis.Exponent = 0;
        low = min(y(:)); high = max(y(:)); span = max(high-low, 0.01);
        if specs{k,4}
            % Preserve a clear upper area for the R0-style inset/legend.
            lower = low - .12*span;
            upper = high + 1.25*span;
            under_inset = y(t >= 2.3,:);
            upper = max(upper, lower + (max(under_inset(:))-lower)/.31);
            ylim(ax, [lower upper]);
        else
            ylim(ax, [low - .10*span, high + .35*span]);
        end
        if specs{k,5}
            % User rule Nov 9: every legend flush to the panel's upper-right
            % corner (applies to all A-series panels; the 'north' branch was
            % retired). Fig 9 does the same on the matplotlib side.
            lg = legend(ax, handles(1:4), names(1:4), 'NumColumns', 4, 'Location', 'northeast', ...
                'FontName', 'Times New Roman', 'FontSize', 14, 'Interpreter', 'latex', 'Box', 'on');
            lg.ItemTokenSize = [20 10];
        else
            lg = legend(ax, handles(1:4), names(1:4), 'Location', 'northeast', ...
                'FontName', 'Times New Roman', 'FontSize', 14, 'Interpreter', 'latex', 'Box', 'on');
            lg.ItemTokenSize = [20 10];
        end
        set(lg, 'EdgeColor', 'black', 'Color', 'white');
        if specs{k,4}
            windows = [0 .035; 1.998 2.035; 8 10];
            range = windows(ci,:);
            insetpos = [pos(1)+.245*pos(3), pos(2)+.48*pos(4), .415*pos(3), .43*pos(4)];
            small = axes(fig, 'Position', insetpos); hold(small, 'on');
            mask = t >= range(1) & t <= range(2);
            for i = 1:4
                plot(small, t(mask), y(mask,i), '-.', 'Color', colors(i,:), 'LineWidth', 1.5);
            end
            if strcmp(specs{k, 2}, 'voltage_deviation')
                plot(small, t(mask), c.voltage_error(mask), 'k--', 'LineWidth', 2);
            end
            set(small, 'FontName', 'Helvetica', 'FontSize', 14, 'LineWidth', .5, ...
                'Box', 'on', 'TickDir', 'in', 'XLim', range, 'XGrid', 'off', 'YGrid', 'off');
            small.XAxis.Exponent = 0;
            small.YAxis.Exponent = 0;
            if ci == 1, xticks(small, [0 .01 .02 .03]);
            elseif ci == 2, xticks(small, [2 2.01 2.02 2.03]);
            else, xticks(small, [8 9 10]); end
            ylocal = y(mask,:); sl = min(ylocal(:)); sh = max(ylocal(:));
            % Floor at a readable decade: windows whose content is machine-flat
            % (Case-1 steady residuals) would otherwise zoom to 1e-6..1e-12 junk.
            ss = max(sh-sl, 1e-2); ylim(small, [sl-.12*ss sh+.12*ss]);
            drawnow;
            ticks = get(small, 'YTick');
            if numel(ticks)>3, yticks(small, ticks([1 round(numel(ticks)/2) end])); end
            % A thin black arrow associates each inset with its true time window.
            target_t = mean(range);
            % Target time lies away from jumps, so interp1 uses unique times.
            [unique_t, unique_indices] = unique(t, 'last');
            target_y = mean(interp1(unique_t, y(unique_indices,:), target_t));
            lim = ylim(ax);
            target = [pos(1)+pos(3)*target_t/10, ...
                      pos(2)+pos(4)*min(.93,max(.07,(target_y-lim(1))/diff(lim)))];
            if ci < 3, tail_x = insetpos(1); else, tail_x = insetpos(1)+insetpos(3); end
            annotation(fig, 'arrow', [target(1) tail_x], [target(2) insetpos(2)], ...
                'Color', 'black', 'LineWidth', .5, 'HeadLength', 7, 'HeadWidth', 7);
        end
    end
    drawnow;
    savefig(fig, fullfile(output, [specs{k,1} '.fig']));
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, fullfile(output, [specs{k,1} '.eps']), '-depsc', '-painters', '-r144');
    exportgraphics(fig, fullfile(output, [specs{k,1} '.png']), ...
        'Resolution', 180, 'BackgroundColor', 'white');
    close(fig);
    fprintf('Exported %s\n', specs{k,1});
end
end
