import torch

# EMA (Exponential Moving Average) für Mean Teacher Modelle 

@torch.no_grad()
def copy_student_to_teacher(student_model, teacher_model):
    teacher_model.load_state_dict(student_model.state_dict())


@torch.no_grad()
def update_teacher(student_model, teacher_model, ema_decay=0.99):
    student_state = student_model.state_dict()
    teacher_state = teacher_model.state_dict()

    for key in teacher_state.keys():
        if teacher_state[key].dtype.is_floating_point:
            if "running_mean" in key or "running_var" in key:
                teacher_state[key].copy_(student_state[key])
            elif "num_batches_tracked" in key:
                teacher_state[key].copy_(student_state[key])
            else:
                teacher_state[key].mul_(ema_decay).add_(student_state[key], alpha=1.0 - ema_decay)
        else:
            teacher_state[key].copy_(student_state[key])