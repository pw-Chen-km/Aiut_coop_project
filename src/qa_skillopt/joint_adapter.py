"""Full QA rollout adapter with gold-blind runtime and routing-only reflection."""
from pathlib import Path
import json
from .adapter import PhaseBatch
from .joint import unpack_workset, content_hash
from .joint_feedback import navigation_feedback, required_scope_sets, aggregate, navigation_cost
from .io import write_new

CONSTRAINTS = '''Improve reusable source navigation across the supplied Markdown workset.
Choose descriptions, information organization and formatting that help the small model
locate appropriate source scopes. Preserve the workset boundaries and registry references.
For failures, infer shared routing ambiguities across information needs. For successes,
look for avoidable calls, unnecessary branches, repeated reading or verbose navigation.
Propose a reusable mechanism hypothesis for each edit; a single-case hypothesis needs
independent validation. No useful improvement is a valid no-edit outcome.
The trajectory supplies routing labels, not answer supervision. Leave evidence content,
source scopes, the tree, models and runtime settings fixed. Preserve original-concept meaning.
Do not embed full questions, IDs of QA examples, answer lookup rules, or instructions to
override the QA runtime. All selected Markdown files are edited and evaluated atomically.
Return the standard SkillOPT patch with reasoning and exact append/insert_after/replace/delete
edits. The edit count and Markdown arrangement are unrestricted within the context budget.
Keep each file's text inside its BEGIN/END pair; extend a file by replacing or inserting
after an exact, unique anchor inside that file. The outer workset is a transport envelope.
'''


def run_batch(rows, *, snapshot, targets, skill_content, session_factory, tree, oracles,
              alternatives, out_dir, phase, judge=None, extra_answer_rows=()):
    edited = {**snapshot, **unpack_workset(skill_content, targets)}
    output = Path(out_dir); output.mkdir(parents=True, exist_ok=True)
    results, answers, views, costs = [], {}, [], {}
    requested = {r['qid'] for r in rows}
    full = list(rows) + [r for r in extra_answer_rows if r['qid'] not in requested]
    with session_factory(navigation_md_overrides=edited) as session:
        for qa in full:
            record = output / 'predictions' / content_hash(qa['qid'])
            record.mkdir(parents=True, exist_ok=False)
            kw = {'qid': qa['qid']}
            if qa.get('history'):
                kw['history'] = qa['history']
            trace = session.agent.answer(qa['question'], **kw)
            write_new(record / 'raw_trajectory.json', trace)
            if phase == 'validation':
                judgment = judge.score(qa, trace)
                if type(judgment.get('hard')) is not int or judgment['hard'] not in (0, 1):
                    raise ValueError('Unavailable answer judgment')
                answers[qa['qid']] = judgment['hard']
                costs[qa['qid']] = navigation_cost(trace)
                write_new(record / 'answer_judgment.json', judgment)
            if qa['qid'] not in requested:
                continue
            feedback = navigation_feedback(qa, trace, oracles[qa['qid']], tree, alternatives[qa['qid']])
            view = feedback.pop('optimizer_view')
            row = {'id': record.name, 'qid': qa['qid'], 'phase': phase,
                   'hard': feedback['route_complete'], 'soft': (feedback['scope_f1'] + feedback['evidence_recall'])/2,
                   'task_type': qa.get('question_type', 'qa'), 'n_turns': trace.get('rounds_used', 0), **feedback}
            results.append(row)
            if phase == 'optimization':
                write_new(record / 'conversation.json', [{'role': 'user', 'content': json.dumps(view, ensure_ascii=False)}])
                views.append(view)
            write_new(record / 'routing_metrics.json', feedback)
    payload = {'results': results, 'hard_accuracy': sum(r['hard'] for r in results)/len(results)}
    if phase == 'validation':
        payload['metrics'] = aggregate(results, answers, costs)
    write_new(output / 'qa_results.json', payload)
    return results, payload.get('metrics'), views


def make_joint_adapter(*, train_rows, validation_rows, snapshot, targets, session_factory,
                       tree, oracles, blocks, judge, metrics):
    from skillopt.envs.base import EnvAdapter
    labels = {p: {q['qid']: required_scope_sets(oracles[p].get(q['qid'], {}), tree, blocks) for q in rows}
              for p, rows in [('optimization', train_rows), ('validation', validation_rows)]}
    eligible_train = [q for q in train_rows if labels['optimization'][q['qid']]]
    eligible_valid = [q for q in validation_rows if labels['validation'][q['qid']]]
    if not eligible_train or not eligible_valid:
        raise ValueError('No known routing labels for training/validation')

    class Adapter(EnvAdapter):
        def setup(self, cfg):
            super().setup(cfg)
            self.analyst_workers, self.failure_only, self.minibatch_size, self.edit_budget = 1, False, 5, 1
            self.cursor = 0
        def build_train_env(self, batch_size, seed, **kw):
            batch = eligible_train[self.cursor:self.cursor+batch_size]; self.cursor += len(batch)
            return PhaseBatch(batch, 'optimization')
        def build_eval_env(self, env_num, split, seed, **kw):
            if split not in ('val', 'valid_seen'):
                raise ValueError('Training cannot read test')
            return PhaseBatch(eligible_valid, 'validation')
        def rollout(self, env_manager, skill_content, out_dir, **kw):
            if not isinstance(env_manager, PhaseBatch) or env_manager.phase not in labels:
                raise ValueError('Unknown phase')
            rows, metric, _ = run_batch(env_manager, snapshot=snapshot, targets=targets,
                skill_content=skill_content, session_factory=session_factory, tree=tree,
                oracles=oracles[env_manager.phase], alternatives=labels[env_manager.phase],
                out_dir=out_dir, phase=env_manager.phase, judge=judge,
                extra_answer_rows=validation_rows if env_manager.phase == 'validation' else ())
            if metric:
                metrics[content_hash(skill_content)] = metric
            return rows
        def reflect(self, results, *args, **kw):
            if any(r.get('phase') != 'optimization' for r in results):
                raise ValueError('Validation reflection forbidden')
            # Only id/hard are consumed from results upstream; conversation.json
            # carries the deliberately sanitized view. Keep other audit fields out.
            clean = [{k: r[k] for k in ('id', 'hard', 'soft', 'task_type', 'n_turns')} for r in results]
            return super().reflect(clean, *args, **kw)
        def get_task_types(self):
            return sorted({q.get('question_type', 'qa') for q in train_rows})
        def get_error_minibatch_prompt(self):
            return CONSTRAINTS
        def get_success_minibatch_prompt(self):
            return CONSTRAINTS
    adapter = Adapter()
    adapter.eligible_train, adapter.labels = eligible_train, labels
    return adapter
