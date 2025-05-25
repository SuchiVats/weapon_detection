from rfdetr.models.rf_detr import build_rf_detr_model as build_rf_detr
from rfdetr.util.misc import NestedTensor  # assuming you use it

def build_model(args):
    model, criterion, postprocessors = build_rf_detr(args)
    return model, criterion, postprocessors
