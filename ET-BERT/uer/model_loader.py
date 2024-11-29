import torch


def load_model(model, model_path):
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    return model

def load_teacher_model(args, model):
    if args.teacher_model_path:
        if hasattr(model, "module"):
            model.module.load_state_dict(torch.load(args.teacher_model_path, map_location="cpu"), strict=False)
        else:
            model.load_state_dict(torch.load(args.teacher_model_path, map_location="cpu"), strict=False)
    return model

def load_from_teacher(model, teacher_model_path):
    '''
    Teacher model has 12 layers where as student model has n layers where 12 % n == 0
    Load Teacher model weights to student model
    '''
    teacher_model = torch.load(teacher_model_path, map_location="cpu")
    student_model = model.state_dict()

    
