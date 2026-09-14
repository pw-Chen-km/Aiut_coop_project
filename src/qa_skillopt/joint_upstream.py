"""Version-checked, scoped compatibility hooks; upstream owns the training loop."""
from contextlib import contextmanager
import re


@contextmanager
def joint_compatibility(trainer, gate):
    from .upstream import verify_loaded_upstream
    verify_loaded_upstream()
    from skillopt.gradient import reflect
    old_rank, old_gate = trainer.rank_and_select, trainer.evaluate_gate
    old_truncate, old_chat = reflect.truncate_payload, reflect.chat_optimizer

    def chat(**kwargs):
        # Remove precisely the pinned upstream numeric editing-budget line.
        kwargs['user'] = re.sub(r'Produce at most L=\d+ [^\n]+\.\n',
            'Propose the necessary edits within the request budget.\n', kwargs['user'])
        return old_chat(**kwargs)

    trainer.rank_and_select = lambda skill_content, patch, **kw: patch
    trainer.evaluate_gate = gate
    reflect.truncate_payload = lambda *args, **kwargs: None
    reflect.chat_optimizer = chat
    try:
        yield
    finally:
        trainer.rank_and_select, trainer.evaluate_gate = old_rank, old_gate
        reflect.truncate_payload, reflect.chat_optimizer = old_truncate, old_chat
