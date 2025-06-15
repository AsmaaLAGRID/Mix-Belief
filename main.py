import os
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from models.trainers import CurriculumTrainer, Trainer
import numpy as np


@hydra.main(version_base="1.1", config_path="configs", config_name="default")
def main(cfg: DictConfig):
    base = hydra.utils.get_original_cwd()
    
    out_dir = os.path.join(base, cfg.output.result_dir, cfg.dataset.name, cfg.experiment)
    os.makedirs(out_dir, exist_ok=True)
    results_file = os.path.join(out_dir, "_results.txt")

    log_file = os.path.join(out_dir, "main.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filemode="a"
    )
    logger = logging.getLogger(__name__)
    logger.info("=== Start Experiment ===")

    results = {}
    for seed in cfg.train.seeds:
        cfg.train.seed = seed
        logger.info(f"--- Run for seed = {seed} ---")

        if cfg.train.curriculum : 
            trainer = CurriculumTrainer(cfg, cfg.train.belief_ratio_threshold)
            logger.info("Curriculum Training Selected !")
        else:
            trainer = Trainer(cfg)
            logger.info("Simple Training Selected !")

        metrics = trainer.run()

        # stocke et loggue
        logger.info(f"Results for seed {seed} : {metrics}")
        results[seed] = metrics

    logger.info("=== End of runs ===")
    save_mean_std_results(results, results_file)


    # tu peux retourner la map seed→metrics si besoin
    return results


def save_mean_std_results(results, result_file):
    # Prépare les listes pour chaque métrique d'intérêt
    test_f1, test_gm, test_u, test_mcce = [], [], [], []

    for seed, metrics in results.items():
        # Adapte les clés selon ce que retourne trainer.run()
        test_f1.append(metrics.get("f1", None))
        test_gm.append(metrics.get("gm", None))
        test_u.append(metrics.get("mean_u", None))
        test_mcce.append(metrics.get("mcce", None))

    # Filtre les valeurs None (au cas où un run échoue)
    test_f1 = [v for v in test_f1 if v is not None]
    test_gm = [v for v in test_gm if v is not None]
    test_u = [v for v in test_u if v is not None]
    test_mcce = [v for v in test_mcce if v is not None]

    with open(result_file, 'a') as f:
        f.write('test f1:' + str(test_f1) + '\n')
        f.write('test gm:' + str(test_gm) + '\n')
        f.write('test uncertainty:' + str(test_u) + '\n')
        f.write('test calibration error:' + str(test_mcce) + '\n')
        if test_gm:
            f.write('mean test gm:' + str(np.mean(test_gm)) + '\n')
            f.write('std test gm:' + str(np.std(test_gm, ddof=1)) + '\n')
        if test_f1:
            f.write('mean test f1:' + str(np.mean(test_f1)) + '\n')
            f.write('std test f1:' + str(np.std(test_f1, ddof=1)) + '\n')
        if test_u:
            f.write('mean test uncertainty:' + str(np.mean(test_u)) + '\n')
            f.write('std test uncertainty:' + str(np.std(test_u, ddof=1)) + '\n')
        if test_mcce:
            f.write('mean test calibration error::' + str(np.mean(test_mcce)) + '\n')
            f.write('std test calibration error::' + str(np.std(test_mcce, ddof=1)) + '\n')
        f.write('\n\n')
    

if __name__ == "__main__":
    results= main()


