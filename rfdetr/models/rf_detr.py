from rfdetr.models.lwdetr import build_lw_detr as build_rf_detr
  # Make sure lwdetr.py exists

def build_rf_detr_model(args):
    return build_rf_detr(args)
