

from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import minimize

class DrydownModel():
    def __init__():
        None

    def fit_drydown_models(self, events):
        """ Fit the drydown models """
        drydown_params = []
        for index, row in event_df_long.iterrows():
            try:
                drydown_params.append(self.calc_drydown(events))
            except Exception as e:
                print(e)
                return None
        drydowns = pd.DataFrame(drydown_params)

        if plot:
            self.plot_drydown_models()
        return pd.merge(events, drydown_params_df, on=['event_start', 'event_end'], how='outer')
        
    def calc_drydown(self, event):
        # Read the data
        start_date = row['event_start']
        end_date = row['event_end']
        delta_theta = row['delta_theta']
        soil_moisture_subset = np.asarray(row['normalized_S'])
        t = np.arange(0, len(soil_moisture_subset),1)
        soil_moisture_range = np.nanmax(soil_moisture_subset) - np.nanmin(soil_moisture_subset)
        soil_moisture_subset_min = np.nanmin(soil_moisture_subset)
        soil_moisture_subset_max = np.nanmax(soil_moisture_subset)
        x = t[~np.isnan(soil_moisture_subset)]
        y = soil_moisture_subset[~np.isnan(soil_moisture_subset)]

        # Define the bounds
        # exp_model(t, delta_theta, theta_w, tau):
        bounds  = [(0, min_sm, 0), (2*soil_moisture_range, soil_moisture_subset_min, np.inf)]
        p0      = [0.5*soil_moisture_range, (soil_moisture_subset_min+min_sm)/2, 1]
        try: 
            # Fit the data
            popt, pcov = curve_fit(f=exp_model, xdata=x, ydata=y, p0=p0, bounds=bounds)
            # popt: Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized
            # pcov: The estimated covariance of popt
            
            # Reroduce the analytical solution and calculate the residuals
            y_opt = exp_model(x, *popt)
            residuals = y - y_opt
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.nanmean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Save
            drydown_params.append({'event_start': start_date, 'event_end': end_date, 'delta_theta': popt[0], 'theta_w': popt[1], 'tau': popt[2], 'r_squared': r_squared, 'opt_drydown': y_opt.tolist()})
            
        except Exception as e:
            print("An error occurred:", e)
            return None

    def fit_drydown_models(self):
        # Convert the event start/end columns to datetime format
        event_df_with_curvefit['event_start'] = pd.to_datetime(event_df_long['event_start'])
        event_df_with_curvefit['event_end'] = pd.to_datetime(event_df_long['event_end'])

        # Determine the number of columns needed for the subplots grid
        num_events = len(event_df_with_curvefit)
        num_cols = 2
        num_rows = int(num_events / num_cols) + int(num_events % num_cols != 0)

        # Plot each row of the event DataFrame as a time series
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, sharey=True, figsize=(10, 5*num_rows))
        for index, row in event_df_with_curvefit.iterrows():

            x = pd.date_range(start=row['event_start'], end=row['event_end'], freq='D')
            y = np.asarray(row['normalized_S'])
            y_opt = np.asarray(row['opt_drydown'])
            y_opt__q = np.asarray(row['q__opt_drydown'])
            y2 = row['PET']
            t = np.arange(0, len(row['normalized_S']),1)
            r_squared = row['r_squared']
            q__r_squared = row['q__r_squared']
            q = row['q__q']
            tau = row['tau']
            try:
                ax_row = int(index / num_cols)
                ax_col = index % num_cols
                axes[ax_row, ax_col].scatter(x, y)
                axes[ax_row, ax_col].plot(x[~np.isnan(y)], y_opt, alpha=.7, label=f'expoential: R2={r_squared:.2f}; tau={tau:.2f}')
                axes[ax_row, ax_col].plot(x[~np.isnan(y)], y_opt__q, alpha=.7, label=f'q model: R2={q__r_squared:.2f}; q={q:.5f})')
                ax2 = axes[ax_row, ax_col].twinx()
                ax2.scatter(x, y2, color='orange', alpha=.5)
                axes[ax_row, ax_col].set_title(f'Event {index}')
                axes[ax_row, ax_col].set_xlabel('Date')
                axes[ax_row, ax_col].set_ylabel('Soil Moisture')
                axes[ax_row, ax_col].set_xlim([row['event_start'], row['event_end']])
                axes[ax_row, ax_col].legend()
                ax2.set_ylim([0, 8])
                ax2.set_ylabel('PET')
                # Rotate the x tick labels
                axes[ax_row, ax_col].tick_params(axis='x', rotation=45)
            except:
                try:
                    ax_row = int(index / num_cols)
                    ax_col = index % num_cols
                    axes[ax_row].scatter(x, y)
                    axes[ax_row].plot(x[~np.isnan(y)], y_opt, alpha=.7, label=f'expoential: R2={r_squared:.2f}; tau={tau:.2f}')
                    axes[ax_row].plot(x[~np.isnan(y)], y_opt__q, alpha=.7, label=f'q model: R2={q__r_squared:.2f}; q={q:.5f})')
                    ax2 = axes[ax_row].twinx()
                    ax2.scatter(x, y2, color='orange', alpha=.5)
                    axes[ax_row].set_title(f'Event {index}')
                    axes[ax_row].set_xlabel('Date')
                    axes[ax_row].set_ylabel('Soil Moisture')
                    axes[ax_row].set_xlim([row['event_start'], row['event_end']])
                    ax2.set_ylim([0, 8])
                    ax2.set_ylabel('PET')
                    # Rotate the x tick labels
                    axes[ax_row].tick_params(axis='x', rotation=45)
                except:
                    continue
                continue

        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(output_dir, f'pt_{EASE_row_index:03d}_{EASE_column_index:03d}_curvefit.png'))



        
def exponential_model(t, delta_theta, theta_w, tau):
    return delta_theta * np.exp(-t/tau) + theta_w

def neg_log_likelihood(params, t, y):
    delta_theta, theta_w, tau, sigma = params
    y_hat = exp_model(t, delta_theta, theta_w, tau)
    residuals = y - y_hat
    ssr = np.sum(residuals ** 2)
    n = len(y)
    sigma2 = ssr / n
    ll = -(n / 2) * np.log(2 * np.pi * sigma2) - (1 / (2 * sigma2)) * ssr
    return -ll

def q_model(t, k, q, delta_theta, theta_star, theta_w):

    s0 = (delta_theta - theta_w)**(1-q)

    a = (1 - q) / ( ( theta_star - theta_w ) ** q )

    return (- k * a * t + s0 ) ** (1/(1-q)) + theta_w
