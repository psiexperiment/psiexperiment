from PyInstaller.utils.hooks import collect_data_files

# Picks up psi/experiment/psi-logo.png (window icon) and any future data
# files added under psi.experiment, without needing per-consumer
# --add-data flags in whatever freezes psi (e.g. cftscal).
datas = collect_data_files('psi.experiment')
