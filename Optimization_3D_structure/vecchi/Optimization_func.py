import numpy as np
from scipy.optimize import minimize

from statistical_test_func import evaluate_simulation

#####################################################################################################################

# FUNCTIONS TO OPTIMIZE THE INPUTS PARAMETERS TO MAKE THE 3D STUCTURE GENERATED REPRESENTATIVE OF THE EXPERIMENTAL IMAGE
# THE PARAMETERS THAT ARE BEING OPTIMIZE ARE:
# MEAN_RADIUS of the log-normal distibution of the pore size and its STD_DEV

#####################################################################################################################
def save_slice_images(array3d, n_slices, folder):
    indices = np.linspace(0, array3d.shape[2] - 1, n_slices, dtype=int)
    for i in indices:
        img = (array3d[:, :, i] * 255 / array3d.max()).astype(np.uint8)
        path = os.path.join(folder, f"slice_{i}.png")
        Image.fromarray(img).save(path)


def optimize_parameters(fixed_params, init_var_params, n_iter=50):
    best_var_params = init_var_params.copy()
    best_score = -np.inf
    
    # Liste di possibili range di variazione (questi range vanno adattati in base alla tua conoscenza)
    mean_radius_range = (np.log(0.4), np.log(0.6))  # in log-space
    std_radius_range = (0.02, 0.5)
    delta1_range = (0.5, 1.0)
    
    for it in range(n_iter):
        # Estrai nuovi valori random all'interno dei range
        cand_params = {
            "mean_radius": np.random.uniform(*mean_radius_range),
            "std_radius": np.random.uniform(*std_radius_range),
            "delta1": np.random.uniform(*delta1_range)
        }
        print(f"Iteration {it+1}: Trying parameters {cand_params}")
        
        # Valuta la simulazione con questi parametri
        score = evaluate_simulation(cand_params, fixed_params, n3D, L, seed, voxel_rule, homogRule, vtkname, fileCoeff, N_SLICES, experimental_images)
        print(f" --> Combined fitness score: {score:.3f}")
        
        # Se la simulazione produce un punteggio migliore, aggiorna i parametri migliori
        if score > best_score:
            best_score = score
            best_var_params = cand_params.copy()
        
        # Se raggiungiamo o superiamo il fitness target, possiamo interrompere l'ottimizzazione
        if best_score >= TARGET_FITNESS:
            print("Target fitness reached!")
            break
        # Scarto relativo
        err = abs(porosity - fixed_params["por_target"]) / fixed_params["por_target"]
        # Trasformalo in un punteggio che va da 1 (err=0) a 0 (err grande)
        por_fitness = max(0.0, 1.0 - err)

    return best_var_params, best_score
    

