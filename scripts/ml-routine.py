import typst 

from ml.training_regression import main as training_main
from ml.predicting import main as predicting_main
from config_reader import config_ml
from logger import get_logger

LOGGER = get_logger(__name__, "ml-routine")
LOGGER.info('Logging timestamps are respect to America/Lima timezone')

def main():
  LOGGER.info("Workflow start")

  training_args = {
    'csv_path': "../data/csv_data/saf_relax-results.csv",
    'model_name': config_ml['model'],
    'epochs': config_ml['epochs'],
    'batch_size': config_ml['batch_size'],
    'lr': 0.001,
    'patience': 50
  }
  
  sys_inputs, parent_folder = training_main(**training_args)

  predicting_args = {
    'model_path': sys_inputs['model-save-path'],
    'DMI': config_ml['DMI_predict'],
    'Ku': config_ml['Ku_predict'],
    'resolution': config_ml['resolution'],
    'task':'regression',
    'save_path': f'{parent_folder}/phase_diagram_prediction.png'
  }

  predicting_main(**predicting_args)

  sys_inputs.update(
    {
      'dmi-value': str(predicting_args['DMI']),
      'ku-value': str(predicting_args['Ku']),
      'pm-resolution': str(predicting_args['resolution']),
      'phase-diagram-img': str(predicting_args['save_path'])
    }
  )

  LOGGER.info(sys_inputs)

  typst.compile(
    input='report_template.typ',
    output=f'{parent_folder}/report.pdf',
    root='..',
    sys_inputs=sys_inputs
  )

if __name__ == '__main__':
  main()