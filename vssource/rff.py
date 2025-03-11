from __future__ import annotations

import warnings
from copy import deepcopy
from itertools import count
from typing import Sequence

from vstools import ConstantFormatVideoNode, CustomRuntimeError, T, flatten, remap_frames, vs

__all__ = [
    'apply_rff_array', 'apply_rff_video',
    'cut_array_on_ranges'
]


def apply_rff_array(old_array: Sequence[T], rff: Sequence[int], tff: Sequence[int], prog_seq: Sequence[int]) -> list[T]:
    array_double_rate = list[T]()

    for prog, arr, rffv, tffv in zip(prog_seq, old_array, rff, tff):
        repeat_amount = (3 if rffv else 2) if prog == 0 else ((6 if tffv else 4) if rffv else 2)

        array_double_rate.extend([arr] * repeat_amount)

    # assert (len(array_double_rate) % 2) == 0
    if (len(array_double_rate) % 2) != 0:
        warnings.warn('uneven amount of fields removing last\n')
        array_double_rate = array_double_rate[:-1]

    # It seems really weird thats its allowed to have rff stuff across
    # vob boundries even for multi angle stuff i have seen this so often though it is ok to remove the warnings
    for i, f1, f2 in zip(count(), array_double_rate[::2], array_double_rate[1::2]):
        if f1 != f2:
            warnings.warn(
                f'Ambiguous pattern due to rff {f1}!={f2} on index {i}\n'
            )

    # 1::2 because there might be the case where there is a one frame cell at the end and if we cut off the last field
    # we will miss that vobid with 0::2
    return array_double_rate[1::2]


def apply_rff_video(
    node: vs.VideoNode, rff: list[int], tff: list[int], prog: list[int], prog_seq: list[int]
) -> ConstantFormatVideoNode:
    assert len(node) == len(rff) == len(tff) == len(prog) == len(prog_seq)

    fields = list[dict[str, int]]()
    tfffs = node.std.RemoveFrameProps(['_FieldBased', '_Field']).std.SeparateFields(True)

    for i, current_prg_seq, current_prg, current_rff, current_tff in zip(count(), prog_seq, prog, rff, tff):
        if not current_prg_seq:
            if current_tff:
                first_field = 2 * i
                second_field = 2 * i + 1
            else:
                first_field = 2 * i + 1
                second_field = 2 * i

            fields += [
                {'n': first_field, 'tf': current_tff, 'prg': False, 'repeat': False},
                {'n': second_field, 'tf': not current_tff, 'prg': False, 'repeat': False}
            ]

            if current_rff:
                assert current_prg
                repeat_field = deepcopy(fields[-2])
                repeat_field['repeat'] = True
                fields.append(repeat_field)
        else:
            assert current_prg

            field_count = 1
            if current_rff:
                field_count += 1 + int(current_tff)

            fields += [
                {'n': 2 * i, 'tf': 1, 'prg': True, 'repeat': False},
                {'n': 2 * i + 1, 'tf': 0, 'prg': True, 'repeat': False}
            ] * field_count

    # TODO: mark known progressive frames as progressive

    # assert (len(fields) % 2) == 0
    if (len(fields) % 2) != 0:
        warnings.warn('uneven amount of fields removing last\n')
        fields = fields[:-1]

    for a, tf, bf in zip(count(), fields[::2], fields[1::2]):
        if tf['tf'] == bf['tf']:
            bf['tf'] = not bf['tf']

            warnings.warn(f'Invalid field transition at {a}')

    for fcurr, fnext in zip(fields[::2], fields[1::2]):
        if fcurr['tf'] == fnext['tf']:
            raise CustomRuntimeError(
                f'Found invalid stream with two consecutive {"top" if fcurr["tf"] else "bottom"} fields!'
            )

    final = remap_frames(tfffs, [x['n'] for x in fields])

    def _set_field(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        f = f.copy()

        f.props.pop('_FieldBased', None)
        f.props._Field = fields[n]['tf']

        return f

    final = vs.core.std.ModifyFrame(final, final, _set_field)

    woven = final.std.DoubleWeave()[::2]

    def _set_repeat(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        f = f.copy()
        if fields[n * 2]['repeat']:
            f.props['RepeatedField'] = 1
        elif fields[n * 2 + 1]['repeat']:
            f.props['RepeatedField'] = 0
        else:
            f.props['RepeatedField'] = -1
        return f

    woven = vs.core.std.ModifyFrame(woven, woven, _set_repeat)

    # TODO: this seems to not work or atleast useless since its disable for non progressive sequence which is rare
    def _update_progressive(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f.copy()

        tf = fields[n * 2]
        bf = fields[n * 2 + 1]

        if tf['prg'] and bf['prg']:
            fout.props['_FieldBased'] = 0

        return fout

    return vs.core.std.ModifyFrame(woven, woven, _update_progressive)


def cut_array_on_ranges(array: list[T], ranges: list[tuple[int, int]]) -> list[T]:
    return [array[i] for i in flatten([range(rrange[0], rrange[1] + 1) for rrange in ranges])]
