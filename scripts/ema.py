import torch
# exponientiell gemitteltes Mittel (EMA) für Teacher-Modelle um den Teacher nach der Back-Propagation des Student-Modells zu aktualisieren 

@torch.no_grad()
def copy_student_to_teacher(student_model, teacher_model):
    """
    Initialisiert den Teacher mit den gleichen Gewichten wie der Student.
    Wird einmal vor dem Training aufgerufen.
    """
    teacher_model.load_state_dict(student_model.state_dict())


@torch.no_grad()
def update_teacher(student_model, teacher_model, ema_decay=0.99):
    """
    EMA Update:
    teacher = ema_decay * teacher + (1 - ema_decay) * student
    """
    for teacher_param, student_param in zip(
        teacher_model.parameters(),
        student_model.parameters()
    ):
        teacher_param.data.mul_(ema_decay).add_(
            student_param.data,
            alpha=1.0 - ema_decay
        )
    