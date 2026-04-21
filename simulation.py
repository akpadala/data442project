import numpy as np
from scipy.integrate import solve_ivp
import dynamics
import parameters

def run_lock_cycle():
    # Initial State: [h_lock, h_b3, h_b2, h_b1, Q]
    y0 = [
        parameters.LOCK_H_UPPER, 
        parameters.BASIN_3_H_INIT, 
        parameters.BASIN_2_H_INIT, 
        parameters.BASIN_1_H_INIT, 
        0.0
    ]
    
    t_total = np.array([])
    x_total = np.empty((5, 0))
    log = []
    
    current_time = 0.0
    # Map stages 1-4 to basin indices (3, 2, 1) and (0 for sea)
    stages = [
        (3, dynamics.WSB_DRAIN_1), 
        (2, dynamics.WSB_DRAIN_2), 
        (1, dynamics.WSB_DRAIN_3), 
        (0, dynamics.FINAL_DRAIN)
    ]
    
    for basin_idx, name in stages:
        t_start = current_time
        
        # Define event: stop when delta_h < EPSILON
        def event_equalize(t, y):
            target = dynamics.H_sea(t) if basin_idx == 0 else y[basin_idx]
            return (y[0] - target) - parameters.EPSILON
        
        event_equalize.terminal = True
        
        # Integration
        sol = solve_ivp(
            fun=lambda t, y: dynamics.system_dynamics(t, y, basin_idx),
            t_span=(t_start, t_start + 3600), # 1 hour max per stage
            y0=y0,
            events=event_equalize,
            method='RK45',
            max_step=parameters.DT_MAX
        )
        
        # Concatenate results
        t_total = np.concatenate([t_total, sol.t])
        x_total = np.hstack([x_total, sol.y])
        
        # Log event
        log.append((name, t_start, sol.t[-1]))
        
        # Prepare next iteration: maintain levels, reset Flow (Q) to 0
        current_time = sol.t[-1]
        y0 = sol.y[:, -1]
        y0[4] = 0.0 # Reset Flow for next stage
        
    return t_total, x_total, log

if __name__ == "__main__":
    # Test run
    t, x, log = run_lock_cycle()
    print(f"Simulation complete. Total time: {t[-1]/60:.2f} minutes.")