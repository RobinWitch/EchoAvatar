# -*- coding: utf-8 -*-
r"""Retarget ZeroEGGS FBX files to AX_female2 and export FBX + BVH.

Recommended command (PowerShell):

    & "D:\MotionBuilder 2026\bin\x64\mobupy.exe" `
      "C:\path\to\motionbuilder_retarget_batch.py" `
      --target-fbx "C:\path\to\AX_female2.fbx" `
      --source-dir "E:\path\to\source_fbx" `
      --source-type zeroeggs `
      --output-dir "E:\path\to\retarget_output" `
      --overwrite

Use --source-file instead of --source-dir to process one FBX.  Existing FBX/BVH
pairs are skipped by default; --overwrite replaces only those output files.

The source characterization stance is reconstructed from the supplied Maya
HumanIK MEL: shoulders Z +/-90, feet X +25, toes X -20, all other local joint
rotations zero, and Hips translation (0, 86.26992798, 0).

AX_female2 is temporarily characterized in a symmetric T-pose by rotating its
two Shoulder_* joints -38.5 degrees around their local Y axes.  This temporary
pose is never saved as the target bind/rest pose.

For ZeroEGGS, a fitted rest-relative correction is applied to the two
ThumbFinger1_* joints after plotting.  MotionBuilder back-solves each corrected
global orientation through the target PreRotation before the local XYZ curves
are keyed, so the correction is present consistently in both FBX and BVH.

The BVH writer removes Global plus names ending in "_scale", and uses Root_M
as the BVH root.  Its offsets are rebuilt from the original target FBX's
world-space rest positions so FBX PreRotation/joint-orient values cannot
distort the BVH zero-rotation rest pose.  Root_M position channels contain
absolute BVH-space positions, matching Blender's BVH importer convention.
"""

from __future__ import print_function

import argparse
import datetime
import io
import json
import math
import os
import sys
import traceback

import pyfbsdk as sdk

if not hasattr(sdk, "FBApplication"):
    import pyfbstandalone

    pyfbstandalone.initialize()


# ---------------------------------------------------------------------------
# Dataset-specific character settings
# ---------------------------------------------------------------------------

SOURCE_NAMESPACE = "SRC"
OUTPUT_SUFFIX = ""
TARGET_TPOSE_SHOULDER_Y = -38.5
# Constant pre/post corrections in the rebased BVH local-rotation space.  They
# were fitted against the established Maya result after removing its Z-up to
# MotionBuilder Y-up basis change.  This preserves the original time-varying
# thumb motion while restoring the intended thumb-root orientation.
TARGET_THUMB1_REBASED_CORRECTION_XYZ = {
    "ThumbFinger1_R": {
        "pre": (-12.299462, 15.428764, 2.271223),
        "post": (34.622728, -19.708258, -33.796142),
    },
    "ThumbFinger1_L": {
        "pre": (-12.618184, -15.491265, -1.717583),
        "post": (34.749653, 20.097533, 33.472851),
    },
}
BVH_ROTATION_ORDER = sdk.FBRotationOrder.kFBXYZ
SOURCE_TYPES = ("zeroeggs", "motorica")
SOURCE_TPOSE_VERTICAL_TOLERANCE = {
    "zeroeggs": 0.1,
    "motorica": 0.6,
}


SOURCE_MAP = {
    "Reference": "Reference",
    "Hips": "Hips",
    "LeftUpLeg": "LeftUpLeg",
    "LeftLeg": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
    "RightUpLeg": "RightUpLeg",
    "RightLeg": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
    "Spine": "Spine",
    "Spine1": "Spine1",
    "Spine2": "Spine2",
    "Spine3": "Spine3",
    "Neck": "Neck",
    "Neck1": "Neck1",
    "Head": "Head",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
}

for _side in ("Left", "Right"):
    for _finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for _index in range(1, 5):
            _slot = "{}Hand{}{}".format(_side, _finger, _index)
            SOURCE_MAP[_slot] = _slot


TARGET_MAP = {
    "Reference": "Global",
    "Hips": "Root_M",
    "LeftUpLeg": "Hip_L",
    "LeftUpLegRoll": "HipPart1_L",
    "LeftLeg": "Knee_L",
    "LeftLegRoll": "KneePart1_L",
    "LeftFoot": "Ankle_L",
    "LeftToeBase": "Toes_L",
    "RightUpLeg": "Hip_R",
    "RightUpLegRoll": "HipPart1_R",
    "RightLeg": "Knee_R",
    "RightLegRoll": "KneePart1_R",
    "RightFoot": "Ankle_R",
    "RightToeBase": "Toes_R",
    "Spine": "Spine1_M",
    "Spine1": "Spine1Part1_M",
    "Spine2": "Chest_M",
    "Neck": "Neck_M",
    "Neck1": "NeckPart1_M",
    "Head": "Head_M",
    "LeftShoulder": "Scapula_L",
    "LeftArm": "Shoulder_L",
    "LeftArmRoll": "ShoulderPart1_L",
    "LeftForeArm": "Elbow_L",
    "LeftForeArmRoll": "ElbowPart1_L",
    "LeftHand": "Wrist_L",
    "RightShoulder": "Scapula_R",
    "RightArm": "Shoulder_R",
    "RightArmRoll": "ShoulderPart1_R",
    "RightForeArm": "Elbow_R",
    "RightForeArmRoll": "ElbowPart1_R",
    "RightHand": "Wrist_R",
    "LeftFingerBase": "Cup_L",
    "RightFingerBase": "Cup_R",
}

for _side, _suffix in (("Left", "L"), ("Right", "R")):
    for _finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for _index in range(1, 5):
            _slot = "{}Hand{}{}".format(_side, _finger, _index)
            TARGET_MAP[_slot] = "{}Finger{}_{}".format(
                _finger, _index, _suffix
            )


def _safe_text(value):
    try:
        return str(value)
    except Exception:
        return "<unprintable>"


def _long_name(component):
    try:
        return _safe_text(component.LongName)
    except Exception:
        return ""


def _short_name(model):
    return _long_name(model).rsplit(":", 1)[-1]


def _is_noninteractive(app):
    try:
        return app.ApplicationState in (
            sdk.FBApplicationState.kFBMobuPy,
            sdk.FBApplicationState.kFBBatch,
        )
    except Exception:
        return False


def _make_dir(path):
    # Multiple mobupy workers share the same output root.  The directory can
    # be created by another worker after isdir() and before makedirs(), so
    # tolerate that one benign race while still surfacing real I/O errors.
    if os.path.isdir(path):
        return
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def _all_scene_models(system):
    result = []
    stack = []
    root = system.SceneRootModel
    if root is not None:
        stack.extend(list(root.Children))
    while stack:
        model = stack.pop()
        result.append(model)
        try:
            stack.extend(list(model.Children))
        except Exception:
            pass
    return result


def _find_model(scene, short_name, namespace=None):
    wanted = (
        "{}:{}".format(namespace, short_name) if namespace else short_name
    )
    model = sdk.FBFindModelByLabelName(wanted)
    if model is not None and _long_name(model) == wanted:
        return model
    for candidate in scene.ModelSkeletons:
        if _long_name(candidate) == wanted:
            return candidate
    raise RuntimeError("Missing skeleton model: " + wanted)


def _characterize(scene, name, mapping, namespace=None):
    character = sdk.FBCharacter(name)
    for slot, bone_name in mapping.items():
        prop = character.PropertyList.Find(slot + "Link")
        if prop is None:
            raise RuntimeError("Unknown MotionBuilder Character slot: " + slot)
        prop.append(_find_model(scene, bone_name, namespace))

    if not character.SetCharacterizeOn(True):
        try:
            detail = _safe_text(character.GetCharacterizeError())
        except Exception:
            detail = ""
        raise RuntimeError(
            "Characterization failed for {}: {}".format(name, detail)
        )
    return character


def _create_source_reference(scene):
    reference = sdk.FBModelNull(SOURCE_NAMESPACE + ":Reference")
    reference.Show = False
    reference.Translation = sdk.FBVector3d(0.0, 0.0, 0.0)
    reference.Rotation = sdk.FBVector3d(0.0, 0.0, 0.0)
    reference.Scaling = sdk.FBVector3d(1.0, 1.0, 1.0)
    _find_model(scene, "Hips", SOURCE_NAMESPACE).Parent = reference
    return reference


def _source_rest_rotation(short_name, source_type):
    # Motorica's FBX local-zero stance is already a valid near-T-pose.  Its
    # small left/right height difference is handled by the type-specific
    # validation tolerance rather than fragile per-joint decimal corrections.
    if source_type == "motorica":
        return sdk.FBVector3d(0.0, 0.0, 0.0)
    if source_type != "zeroeggs":
        raise RuntimeError("Unsupported source type: " + source_type)

    if short_name == "RightShoulder":
        return sdk.FBVector3d(0.0, 0.0, 90.0)
    if short_name == "LeftShoulder":
        return sdk.FBVector3d(0.0, 0.0, -90.0)
    if short_name in ("RightFoot", "LeftFoot"):
        return sdk.FBVector3d(25.0, 0.0, 0.0)
    if short_name in ("RightToeBase", "LeftToeBase"):
        return sdk.FBVector3d(-20.0, 0.0, 0.0)
    return sdk.FBVector3d(0.0, 0.0, 0.0)


def _make_characterization_take(
    system, source_take, source_models, source_type
):
    """Build disposable source/target T-poses for characterization only."""
    scene = system.Scene
    player = sdk.FBPlayerControl()
    frame_zero = sdk.FBTime(0, 0, 0, 0)
    target_shoulders = [
        _find_model(scene, "Shoulder_R"),
        _find_model(scene, "Shoulder_L"),
    ]
    system.CurrentTake = source_take
    player.Goto(frame_zero)
    scene.Evaluate()
    local_translations = {
        _long_name(model): sdk.FBVector3d(
            float(model.Translation[0]),
            float(model.Translation[1]),
            float(model.Translation[2]),
        )
        for model in source_models
    }
    # CopyTake is the Autodesk workaround for reliably registering a new take.
    rest_take = source_take.CopyTake("__CHARACTERIZE_REST__")
    if rest_take is None:
        raise RuntimeError("Could not create the characterization take")
    system.CurrentTake = rest_take
    player.Goto(frame_zero)
    if not rest_take.DeleteAnimationOnObjects(
        list(source_models) + target_shoulders
    ):
        raise RuntimeError("Could not clear the characterization take")

    for model in source_models:
        name = _short_name(model)
        translation = local_translations[_long_name(model)]
        if name == "Hips":
            translation = sdk.FBVector3d(0.0, 86.26992798, 0.0)

        model.Translation.SetAnimated(True)
        model.Translation = translation
        model.Translation.KeyAt(frame_zero)

        model.Rotation.SetAnimated(True)
        model.Rotation = _source_rest_rotation(name, source_type)
        model.Rotation.KeyAt(frame_zero)

    # AX_female2's authored rest stance is an A-pose.  Both shoulder joints
    # use the same local-Y direction, so -38.5 degrees raises the two arms into
    # a symmetric T-pose.  This lives only in __CHARACTERIZE_REST__.
    for model in target_shoulders:
        model.Rotation.SetAnimated(True)
        model.Rotation = sdk.FBVector3d(
            0.0, TARGET_TPOSE_SHOULDER_Y, 0.0
        )
        model.Rotation.KeyAt(frame_zero)

    player.Goto(frame_zero)
    scene.Evaluate()
    return rest_take


def _global_position(model):
    value = sdk.FBVector3d()
    model.GetVector(
        value,
        sdk.FBModelTransformationType.kModelTranslation,
        True,
    )
    return tuple(float(value[index]) for index in range(3))


def _copy_matrix(source):
    result = sdk.FBMatrix()
    for index in range(16):
        result[index] = float(source[index])
    return result


def _global_rotation_matrix(model):
    result = sdk.FBMatrix()
    model.GetMatrix(
        result,
        sdk.FBModelTransformationType.kModelRotation,
        True,
    )
    return result


def _matrix_inverse(source):
    result = sdk.FBMatrix()
    sdk.FBMatrixInverse(result, source)
    return result


def _matrix_mult(left, right):
    result = sdk.FBMatrix()
    sdk.FBMatrixMult(result, left, right)
    return result


def _rotation_matrix_xyz(values):
    result = sdk.FBMatrix()
    sdk.FBRotationToMatrix(
        result,
        sdk.FBVector3d(*values),
        sdk.FBRotationOrder.kFBXYZ,
    )
    return result


def _matrix_rotate_vector(matrix, value):
    source = sdk.FBVector4d(value[0], value[1], value[2], 0.0)
    result = sdk.FBVector4d()
    sdk.FBVectorMatrixMult(result, matrix, source)
    return tuple(float(result[index]) for index in range(3))


def _matrix_to_bvh_euler(matrix):
    value = sdk.FBVector3d()
    sdk.FBMatrixToRotation(value, matrix, BVH_ROTATION_ORDER)
    # kFBXYZ returns X/Y/Z.  BVH channels are emitted as Z/Y/X, which gives
    # the standard Rz * Ry * Rx composition when these values are reversed.
    return [float(value[2]), float(value[1]), float(value[0])]


def _vector_add(left, right):
    return tuple(left[index] + right[index] for index in range(3))


def _vector_sub(left, right):
    return tuple(left[index] - right[index] for index in range(3))


def _distance(left, right):
    return math.sqrt(
        sum((left[index] - right[index]) ** 2 for index in range(3))
    )


def _build_preorder(root_name, children):
    result = []

    def visit(name):
        result.append(name)
        for child in children[name]:
            visit(child)

    visit(root_name)
    return result


def _capture_bvh_rest_skeleton(system):
    """Capture the original target FBX rest pose before source append/plot."""
    scene = system.Scene
    player = sdk.FBPlayerControl()
    player.Goto(sdk.FBTime(0, 0, 0, 0))
    scene.Evaluate()

    included = []
    excluded_scale_bones = []
    excluded_global_bone = False
    all_names = set()
    for model in scene.ModelSkeletons:
        name = _short_name(model)
        if name in all_names:
            raise RuntimeError("Duplicate target skeleton name: " + name)
        all_names.add(name)
        if name == "Global":
            excluded_global_bone = True
        elif name.lower().endswith("_scale"):
            excluded_scale_bones.append(name)
        else:
            included.append(model)

    models = {_short_name(model): model for model in included}
    rest_positions = {
        name: _global_position(model) for name, model in models.items()
    }
    rest_rotations = {
        name: _copy_matrix(_global_rotation_matrix(model))
        for name, model in models.items()
    }
    rest_rotation_inverses = {
        name: _matrix_inverse(matrix)
        for name, matrix in rest_rotations.items()
    }

    parents = {}
    children = {name: [] for name in models}
    for name, model in models.items():
        parent = model.Parent
        parent_name = None
        while parent is not None:
            candidate = _short_name(parent)
            if candidate in models:
                parent_name = candidate
                break
            parent = parent.Parent
        parents[name] = parent_name
        if parent_name is not None:
            children[parent_name].append(name)

    roots = [name for name, parent in parents.items() if parent is None]
    if roots != ["Root_M"]:
        raise RuntimeError(
            "Expected one retained BVH root named Root_M, found: {}".format(
                roots
            )
        )
    if not excluded_global_bone:
        raise RuntimeError("The target skeleton does not contain Global")

    preorder = _build_preorder("Root_M", children)
    if len(preorder) != len(models):
        missing = sorted(set(models) - set(preorder))
        raise RuntimeError(
            "Retained BVH hierarchy is disconnected: {}".format(missing[:20])
        )

    offsets = {}
    for name in preorder:
        parent = parents[name]
        offsets[name] = (
            rest_positions[name]
            if parent is None
            else _vector_sub(rest_positions[name], rest_positions[parent])
        )

    return {
        "root": "Root_M",
        "preorder": preorder,
        "parents": parents,
        "children": children,
        "offsets": offsets,
        "rest_positions": rest_positions,
        "rest_rotations": rest_rotations,
        "rest_rotation_inverses": rest_rotation_inverses,
        "excluded_scale_bones": sorted(excluded_scale_bones),
        "excluded_global_bone": excluded_global_bone,
        "source_skeleton_count": len(scene.ModelSkeletons),
    }


def _validate_source_t_pose(scene, vertical_tolerance):
    """Force hierarchy evaluation and verify a real, symmetric T-pose."""
    player = sdk.FBPlayerControl()
    frame_zero = sdk.FBTime(0, 0, 0, 0)
    metrics = None
    for _attempt in range(3):
        # Goto forces MotionBuilder to invalidate stale hierarchy matrices.
        # A plain Evaluate can occasionally reuse a partially updated matrix
        # immediately after CopyTake/DeleteAnimationOnObjects.
        player.Goto(frame_zero)
        scene.Evaluate()
        right_shoulder = _global_position(
            _find_model(scene, "RightShoulder", SOURCE_NAMESPACE)
        )
        right_arm = _global_position(
            _find_model(scene, "RightArm", SOURCE_NAMESPACE)
        )
        right_forearm = _global_position(
            _find_model(scene, "RightForeArm", SOURCE_NAMESPACE)
        )
        left_shoulder = _global_position(
            _find_model(scene, "LeftShoulder", SOURCE_NAMESPACE)
        )
        left_arm = _global_position(
            _find_model(scene, "LeftArm", SOURCE_NAMESPACE)
        )
        left_forearm = _global_position(
            _find_model(scene, "LeftForeArm", SOURCE_NAMESPACE)
        )

        metrics = {
            "RightShoulder": right_shoulder,
            "RightArm": right_arm,
            "RightForeArm": right_forearm,
            "LeftShoulder": left_shoulder,
            "LeftArm": left_arm,
            "LeftForeArm": left_forearm,
        }
        horizontal = all(
            abs(point[1] - right_shoulder[1]) < vertical_tolerance
            for point in (right_arm, right_forearm)
        ) and all(
            abs(point[1] - left_shoulder[1]) < vertical_tolerance
            for point in (left_arm, left_forearm)
        )
        outward = (
            right_forearm[0] < right_arm[0] < right_shoulder[0]
            and left_forearm[0] > left_arm[0] > left_shoulder[0]
        )
        symmetric_height = (
            abs(right_shoulder[1] - left_shoulder[1])
            < vertical_tolerance
        )
        if horizontal and outward and symmetric_height:
            return metrics

    raise RuntimeError(
        "The source characterization pose is not a symmetric T-pose: {}".format(
            metrics
        )
    )


def _validate_target_t_pose(scene):
    """Verify that AX_female2's temporary stance has horizontal arms."""
    metrics = None
    for _attempt in range(3):
        scene.Evaluate()
        right_shoulder = _global_position(_find_model(scene, "Shoulder_R"))
        right_elbow = _global_position(_find_model(scene, "Elbow_R"))
        right_wrist = _global_position(_find_model(scene, "Wrist_R"))
        left_shoulder = _global_position(_find_model(scene, "Shoulder_L"))
        left_elbow = _global_position(_find_model(scene, "Elbow_L"))
        left_wrist = _global_position(_find_model(scene, "Wrist_L"))

        metrics = {
            "Shoulder_R": right_shoulder,
            "Elbow_R": right_elbow,
            "Wrist_R": right_wrist,
            "Shoulder_L": left_shoulder,
            "Elbow_L": left_elbow,
            "Wrist_L": left_wrist,
        }
        horizontal = all(
            abs(point[1] - right_shoulder[1]) < 0.25
            for point in (right_elbow, right_wrist)
        ) and all(
            abs(point[1] - left_shoulder[1]) < 0.25
            for point in (left_elbow, left_wrist)
        )
        outward = (
            right_wrist[0] < right_elbow[0] < right_shoulder[0]
            and left_wrist[0] > left_elbow[0] > left_shoulder[0]
        )
        symmetric = (
            abs(right_shoulder[1] - left_shoulder[1]) < 0.25
            and abs(abs(right_wrist[0]) - abs(left_wrist[0])) < 1.0
        )
        if horizontal and outward and symmetric:
            return metrics

    raise RuntimeError(
        "The target characterization pose is not a symmetric T-pose: {}".format(
            metrics
        )
    )


def _make_plot_options():
    options = sdk.FBPlotOptions()
    options.ConstantKeyReducerKeepOneKey = True
    options.PlotAllTakes = False
    options.PlotOnFrame = True
    options.PlotPeriod = sdk.FBTime(0, 0, 0, 1)
    options.PlotTranslationOnRootOnly = True
    options.PreciseTimeDiscontinuities = True
    options.RotationFilterToApply = (
        sdk.FBRotationFilter.kFBRotationFilterGimbleKiller
    )
    options.UseConstantKeyReducer = True
    return options


def _apply_thumb_root_rebased_correction(
    system, take, snapshot, expected_fps
):
    """Correct ThumbFinger1 in the same rest-relative space used by BVH.

    Applying a raw Euler offset to ThumbFinger1 is incorrect because the AX
    skeleton stores a large PreRotation on that joint.  Instead, reconstruct
    the rest-relative parent/child delta, apply the fitted pre/post matrices,
    convert back to the joint's animated global rotation, and key the solved
    local Euler values.  This updates both the exported FBX and BVH.
    """
    scene = system.Scene
    player = sdk.FBPlayerControl()
    system.CurrentTake = take
    actual_fps = float(player.GetTransportFpsValue())
    if abs(actual_fps - expected_fps) > 0.001:
        raise RuntimeError(
            "Transport FPS changed from {} to {} before thumb correction".format(
                expected_fps, actual_fps
            )
        )

    models = {}
    corrections = {}
    for name, values in TARGET_THUMB1_REBASED_CORRECTION_XYZ.items():
        if name not in snapshot["parents"]:
            raise RuntimeError("Thumb correction joint is absent: " + name)
        parent_name = snapshot["parents"][name]
        if parent_name is None:
            raise RuntimeError("Thumb correction joint has no parent: " + name)
        models[name] = _find_model(scene, name)
        models[parent_name] = _find_model(scene, parent_name)
        corrections[name] = {
            "parent": parent_name,
            "pre": _rotation_matrix_xyz(values["pre"]),
            "post": _rotation_matrix_xyz(values["post"]),
        }

    start_seconds = float(take.LocalTimeSpan.GetStart().GetSecondDouble())
    stop_seconds = float(take.LocalTimeSpan.GetStop().GetSecondDouble())
    frame_count = int(round((stop_seconds - start_seconds) * expected_fps)) + 1
    if frame_count < 2:
        raise RuntimeError("The take is too short for thumb correction")

    corrected_global_rotations = {name: [] for name in corrections}
    corrected_local_deltas = {name: [] for name in corrections}
    # Capture every corrected matrix before editing any curve.  This prevents
    # newly inserted earlier keys from changing later samples through spline
    # interpolation or constant-key reduction.
    for frame_index in range(frame_count):
        sample_time = sdk.FBTime()
        sample_time.SetSecondDouble(
            start_seconds + frame_index / expected_fps
        )
        player.Goto(sample_time)
        scene.Evaluate()
        for name, correction in corrections.items():
            parent_name = correction["parent"]
            parent_delta = _matrix_mult(
                _global_rotation_matrix(models[parent_name]),
                snapshot["rest_rotation_inverses"][parent_name],
            )
            child_delta = _matrix_mult(
                _global_rotation_matrix(models[name]),
                snapshot["rest_rotation_inverses"][name],
            )
            local_delta = _matrix_mult(
                _matrix_inverse(parent_delta), child_delta
            )
            corrected_local = _matrix_mult(
                _matrix_mult(correction["pre"], local_delta),
                correction["post"],
            )
            corrected_child_delta = _matrix_mult(
                parent_delta, corrected_local
            )
            corrected_global = _matrix_mult(
                corrected_child_delta, snapshot["rest_rotations"][name]
            )
            corrected_local_deltas[name].append(
                _copy_matrix(corrected_local)
            )
            corrected_global_rotations[name].append(
                _copy_matrix(corrected_global)
            )

    if not take.DeleteAnimationOnObjects(
        [models[name] for name in sorted(corrections)]
    ):
        raise RuntimeError("Could not clear plotted thumb-root curves")
    solved_eulers = {name: [] for name in corrections}
    previous_eulers = {name: None for name in corrections}
    immediate_global_error = 0.0
    for name in corrections:
        models[name].Rotation.SetAnimated(False)

    # MotionBuilder's FBModel.SetMatrix does not back-solve Rotation curves on
    # these skeleton joints.  SetVector with global_info=True does: it accounts
    # for the joint's PreRotation and current animated parent, then writes the
    # solved local XYZ value to model.Rotation.  Keep the thumbs temporarily
    # unanimated while collecting all solved values, then key them in one pass.
    for frame_index in range(frame_count):
        sample_time = sdk.FBTime()
        sample_time.SetSecondDouble(
            start_seconds + frame_index / expected_fps
        )
        player.Goto(sample_time)
        scene.Evaluate()
        for name in corrections:
            desired_global_euler = sdk.FBVector3d()
            sdk.FBMatrixToRotation(
                desired_global_euler,
                corrected_global_rotations[name][frame_index],
                sdk.FBRotationOrder.kFBXYZ,
            )
            models[name].SetVector(
                desired_global_euler,
                sdk.FBModelTransformationType.kModelRotation,
                True,
            )
        # One hierarchy evaluation resolves both independent thumb roots.
        scene.Evaluate()
        for name in corrections:
            solved_global = _global_rotation_matrix(models[name])
            immediate_global_error = max(
                immediate_global_error,
                max(
                    abs(
                        float(solved_global[index])
                        - float(
                            corrected_global_rotations[name][frame_index][
                                index
                            ]
                        )
                    )
                    for index in range(16)
                ),
            )
            solved_euler = [
                float(models[name].Rotation[index]) for index in range(3)
            ]
            solved_euler = _unwrap_euler(
                solved_euler, previous_eulers[name]
            )
            previous_eulers[name] = list(solved_euler)
            solved_eulers[name].append(solved_euler)

    rotation_nodes = {}
    for name in corrections:
        models[name].Rotation.SetAnimated(True)
        rotation_nodes[name] = models[name].Rotation.GetAnimationNode()
        if rotation_nodes[name] is None:
            raise RuntimeError("Could not animate thumb correction: " + name)
    for frame_index in range(frame_count):
        sample_time = sdk.FBTime()
        sample_time.SetSecondDouble(
            start_seconds + frame_index / expected_fps
        )
        for name in corrections:
            rotation_nodes[name].KeyAdd(
                sample_time, solved_eulers[name][frame_index]
            )
    scene.Evaluate()

    sample_indices = set((0, frame_count // 2, frame_count - 1))
    sampled_matrix_error = 0.0
    sampled_global_error = 0.0
    sampled_raw_local_error = 0.0
    for frame_index in sorted(sample_indices):
        sample_time = sdk.FBTime()
        sample_time.SetSecondDouble(
            start_seconds + frame_index / expected_fps
        )
        player.Goto(sample_time)
        scene.Evaluate()
        for name, correction in corrections.items():
            actual_global = _global_rotation_matrix(models[name])
            sampled_global_error = max(
                sampled_global_error,
                max(
                    abs(
                        float(actual_global[index])
                        - float(
                            corrected_global_rotations[name][frame_index][
                                index
                            ]
                        )
                    )
                    for index in range(16)
                ),
            )
            actual_raw_local = sdk.FBMatrix()
            models[name].GetMatrix(
                actual_raw_local,
                sdk.FBModelTransformationType.kModelRotation,
                False,
            )
            expected_raw_local = _rotation_matrix_xyz(
                solved_eulers[name][frame_index]
            )
            sampled_raw_local_error = max(
                sampled_raw_local_error,
                max(
                    abs(
                        float(actual_raw_local[index])
                        - float(expected_raw_local[index])
                    )
                    for index in range(16)
                ),
            )
            parent_name = correction["parent"]
            parent_delta = _matrix_mult(
                _global_rotation_matrix(models[parent_name]),
                snapshot["rest_rotation_inverses"][parent_name],
            )
            child_delta = _matrix_mult(
                _global_rotation_matrix(models[name]),
                snapshot["rest_rotation_inverses"][name],
            )
            actual_local = _matrix_mult(
                _matrix_inverse(parent_delta), child_delta
            )
            expected_local = corrected_local_deltas[name][frame_index]
            sampled_matrix_error = max(
                sampled_matrix_error,
                max(
                    abs(
                        float(actual_local[index])
                        - float(expected_local[index])
                    )
                    for index in range(16)
                ),
            )

    if sampled_matrix_error > 0.00001:
        raise RuntimeError(
            "Thumb correction matrix error is too large: {} "
            "(immediate_global={}, keyed_global={}, keyed_raw_local={})".format(
                sampled_matrix_error,
                immediate_global_error,
                sampled_global_error,
                sampled_raw_local_error,
            )
        )
    start_time = sdk.FBTime()
    start_time.SetSecondDouble(start_seconds)
    player.Goto(start_time)
    scene.Evaluate()
    return {
        "joints": sorted(corrections),
        "frames": frame_count,
        "sampled_matrix_max_error": sampled_matrix_error,
        "immediate_global_max_error": immediate_global_error,
        "sampled_global_max_error": sampled_global_error,
        "sampled_raw_local_max_error": sampled_raw_local_error,
        "rebased_pre_post_xyz": TARGET_THUMB1_REBASED_CORRECTION_XYZ,
    }


def _key_count(prop):
    node = prop.GetAnimationNode()
    if node is None:
        return 0
    total = 0
    for axis in node.Nodes:
        if axis.FCurve is not None:
            total += len(axis.FCurve.Keys)
    return total


def _clear_target_skeleton_animation(system):
    """Make the target skeleton static at its authored frame-zero pose.

    AX_female2 contains a short setup/test animation on helper joints such as
    Heel_L and Heel_R.  HumanIK plotting overwrites characterized joints, but
    leaves unmapped helper-joint curves untouched.  Those residual curves then
    leak into both the exported FBX and the custom BVH.  Capture frame zero,
    clear the target skeleton's animation in the current take, and restore the
    captured local transforms before the source animation is appended.
    """
    scene = system.Scene
    take = system.CurrentTake
    if take is None:
        raise RuntimeError("The target FBX has no current take")

    player = sdk.FBPlayerControl()
    frame_zero = sdk.FBTime(0, 0, 0, 0)
    player.Goto(frame_zero)
    scene.Evaluate()

    skeletons = list(scene.ModelSkeletons)
    if not skeletons:
        raise RuntimeError("The target FBX contains no skeleton models")

    captured = []
    preexisting_key_count = 0
    animated_models = []
    for model in skeletons:
        transform_key_count = sum(
            _key_count(prop)
            for prop in (model.Translation, model.Rotation, model.Scaling)
        )
        preexisting_key_count += transform_key_count
        if transform_key_count:
            animated_models.append(_long_name(model))
        captured.append(
            (
                model,
                tuple(float(model.Translation[index]) for index in range(3)),
                tuple(float(model.Rotation[index]) for index in range(3)),
                tuple(float(model.Scaling[index]) for index in range(3)),
                _global_position(model),
                _copy_matrix(_global_rotation_matrix(model)),
            )
        )

    if not take.DeleteAnimationOnObjects(skeletons):
        raise RuntimeError("Could not clear animation from the target skeleton")

    for model, translation, rotation, scaling, _, _ in captured:
        model.Translation.SetAnimated(False)
        model.Rotation.SetAnimated(False)
        model.Scaling.SetAnimated(False)
        model.Translation = sdk.FBVector3d(*translation)
        model.Rotation = sdk.FBVector3d(*rotation)
        model.Scaling = sdk.FBVector3d(*scaling)

    player.Goto(frame_zero)
    scene.Evaluate()

    remaining_key_count = 0
    local_max_error = 0.0
    global_position_max_error = 0.0
    global_rotation_matrix_max_error = 0.0
    for (
        model,
        translation,
        rotation,
        scaling,
        global_position,
        global_rotation,
    ) in captured:
        remaining_key_count += sum(
            _key_count(prop)
            for prop in (model.Translation, model.Rotation, model.Scaling)
        )
        for prop, expected in (
            (model.Translation, translation),
            (model.Rotation, rotation),
            (model.Scaling, scaling),
        ):
            for index in range(3):
                local_max_error = max(
                    local_max_error,
                    abs(float(prop[index]) - expected[index]),
                )
        global_position_max_error = max(
            global_position_max_error,
            _distance(_global_position(model), global_position),
        )
        restored_rotation = _global_rotation_matrix(model)
        global_rotation_matrix_max_error = max(
            global_rotation_matrix_max_error,
            max(
                abs(float(restored_rotation[index]) - global_rotation[index])
                for index in range(16)
            ),
        )

    if remaining_key_count:
        raise RuntimeError(
            "Target skeleton still has {} transform keys after cleanup".format(
                remaining_key_count
            )
        )
    if (
        local_max_error > 0.000001
        or global_position_max_error > 0.00001
        or global_rotation_matrix_max_error > 0.00001
    ):
        raise RuntimeError(
            "Target frame-zero pose changed during animation cleanup "
            "(local={}, global_position={}, global_rotation={})".format(
                local_max_error,
                global_position_max_error,
                global_rotation_matrix_max_error,
            )
        )

    return {
        "target_skeleton_count": len(skeletons),
        "preexisting_transform_keys": preexisting_key_count,
        "animated_target_skeletons": sorted(animated_models),
        "remaining_transform_keys": remaining_key_count,
        "frame0_local_max_error": local_max_error,
        "frame0_global_position_max_error": global_position_max_error,
        "frame0_global_rotation_matrix_max_error": (
            global_rotation_matrix_max_error
        ),
    }


def _validate_baked_animation(scene):
    root = _find_model(scene, "Root_M")
    elbow = _find_model(scene, "Elbow_L")
    root_keys = _key_count(root.Translation) + _key_count(root.Rotation)
    elbow_keys = _key_count(elbow.Rotation)
    if root_keys < 10 or elbow_keys < 10:
        raise RuntimeError(
            "Retarget plot produced too few keys (Root_M={}, Elbow_L={})".format(
                root_keys, elbow_keys
            )
        )
    return {"Root_M": root_keys, "Elbow_L": elbow_keys}


def _sample_retargeted_arms(system, take, fps):
    """Record representative arm positions for inspection/reporting."""
    scene = system.Scene
    player = sdk.FBPlayerControl()
    start = float(take.LocalTimeSpan.GetStart().GetSecondDouble())
    stop = float(take.LocalTimeSpan.GetStop().GetSecondDouble())
    frame_count = int(round((stop - start) * fps)) + 1
    sample_indices = sorted(
        set((0, min(100, frame_count - 1), min(500, frame_count - 1), frame_count - 1))
    )
    result = []
    for frame_index in sample_indices:
        sample_time = sdk.FBTime()
        sample_time.SetSecondDouble(start + frame_index / fps)
        player.Goto(sample_time)
        scene.Evaluate()

        root = _global_position(_find_model(scene, "Root_M"))
        elbow_left = _global_position(_find_model(scene, "Elbow_L"))
        elbow_right = _global_position(_find_model(scene, "Elbow_R"))
        wrist_left = _global_position(_find_model(scene, "Wrist_L"))
        wrist_right = _global_position(_find_model(scene, "Wrist_R"))
        valid = (
            elbow_left[0] > root[0] + 5.0
            and elbow_right[0] < root[0] - 5.0
        )
        metrics = {
            "frame": frame_index,
            "Root_M": root,
            "Elbow_L": elbow_left,
            "Wrist_L": wrist_left,
            "Elbow_R": elbow_right,
            "Wrist_R": wrist_right,
            "elbows_on_expected_sides": valid,
        }
        result.append(metrics)
    return result


def _bvh_number(value):
    if abs(value) < 0.00000000005:
        value = 0.0
    return "{:.10g}".format(value)


def _write_bvh_joint(handle, name, children, offsets, depth, is_root=False):
    indent = "  " * depth
    handle.write(indent + ("ROOT " if is_root else "JOINT ") + name + "\n")
    handle.write(indent + "{\n")
    handle.write(
        indent
        + "  OFFSET {} {} {}\n".format(
            *[_bvh_number(value) for value in offsets[name]]
        )
    )
    if is_root:
        handle.write(
            indent
            + "  CHANNELS 6 Xposition Yposition Zposition "
            + "Zrotation Yrotation Xrotation\n"
        )
    else:
        handle.write(
            indent + "  CHANNELS 3 Zrotation Yrotation Xrotation\n"
        )
    for child in children[name]:
        _write_bvh_joint(
            handle, child, children, offsets, depth + 1, False
        )
    if not children[name]:
        handle.write(indent + "  End Site\n")
        handle.write(indent + "  {\n")
        handle.write(indent + "    OFFSET 0 0 0\n")
        handle.write(indent + "  }\n")
    handle.write(indent + "}\n")


def _unwrap_euler(current, previous):
    if previous is None:
        return current
    for index in range(3):
        while current[index] - previous[index] > 180.0:
            current[index] -= 360.0
        while current[index] - previous[index] < -180.0:
            current[index] += 360.0
    return current


def _has_rotation_keys(model):
    node = model.Rotation.GetAnimationNode()
    if node is None:
        return False
    return any(
        axis.FCurve is not None and len(axis.FCurve.Keys) > 0
        for axis in node.Nodes
    )


def _quantized_rest_pose_error(snapshot):
    reconstructed = {}
    errors = []
    for name in snapshot["preorder"]:
        offset = tuple(
            float(_bvh_number(value)) for value in snapshot["offsets"][name]
        )
        parent = snapshot["parents"][name]
        reconstructed[name] = (
            offset
            if parent is None
            else _vector_add(reconstructed[parent], offset)
        )
        errors.append(
            _distance(reconstructed[name], snapshot["rest_positions"][name])
        )
    return max(errors) if errors else 0.0


def _validate_bvh(
    output_path,
    expected_root,
    expected_frames,
    expected_frame_time,
    expected_joint_count,
):
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Expected BVH was not created: " + output_path)

    root_name = None
    joint_count = 0
    frame_count = None
    frame_time = None
    with io.open(output_path, "r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("ROOT "):
                root_name = stripped.split(None, 1)[1]
                joint_count += 1
            elif stripped.startswith("JOINT "):
                joint_count += 1
            elif stripped.startswith("Frames:"):
                frame_count = int(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Frame Time:"):
                frame_time = float(stripped.split(":", 1)[1].strip())
                break

    if root_name != expected_root:
        raise RuntimeError(
            "BVH root is {}, expected {}".format(root_name, expected_root)
        )
    if joint_count != expected_joint_count:
        raise RuntimeError(
            "BVH contains {} joints, expected {}".format(
                joint_count, expected_joint_count
            )
        )
    if frame_count != expected_frames:
        raise RuntimeError(
            "BVH contains {} frames, expected {}".format(
                frame_count, expected_frames
            )
        )
    if frame_time is None or abs(frame_time - expected_frame_time) > 0.0000001:
        raise RuntimeError(
            "BVH frame time {} does not match {}".format(
                frame_time, expected_frame_time
            )
        )
    return frame_count, frame_time, joint_count


def _export_bvh_rebased(
    system,
    take,
    snapshot,
    output_path,
    expected_fps,
):
    """Write a BVH whose zero-rotation pose matches the target FBX.

    FBX joints can store bone direction in PreRotation/joint-orient.  BVH has
    no equivalent field, so exporting the FBX local translations as OFFSET
    produces a visibly different rest skeleton.  Here every OFFSET is rebuilt
    as a parent-to-child vector in the original target's world rest pose, and
    every animated global orientation is rebased against that same rest pose.
    """
    scene = system.Scene
    player = sdk.FBPlayerControl()
    system.CurrentTake = take
    actual_fps = float(player.GetTransportFpsValue())
    if abs(actual_fps - expected_fps) > 0.001:
        raise RuntimeError(
            "Transport FPS changed from {} to {}".format(
                expected_fps, actual_fps
            )
        )

    preorder = snapshot["preorder"]
    wanted = set(preorder)
    animated_models = {}
    for model in scene.ModelSkeletons:
        long_name = _long_name(model)
        if long_name in wanted:
            animated_models[long_name] = model
    missing = sorted(wanted - set(animated_models))
    if missing:
        raise RuntimeError(
            "Retarget scene is missing BVH joints: {}".format(missing[:20])
        )

    animated_rotation = {
        name: _has_rotation_keys(animated_models[name]) for name in preorder
    }
    start_seconds = float(take.LocalTimeSpan.GetStart().GetSecondDouble())
    stop_seconds = float(take.LocalTimeSpan.GetStop().GetSecondDouble())
    frame_count = int(round((stop_seconds - start_seconds) * expected_fps)) + 1
    if frame_count < 2:
        raise RuntimeError("The take is too short to export as BVH")
    frame_time = 1.0 / expected_fps

    rest_error = _quantized_rest_pose_error(snapshot)
    if rest_error > 0.001:
        raise RuntimeError(
            "Rebased BVH rest pose error is too large: {}".format(rest_error)
        )

    root_name = snapshot["root"]
    previous_eulers = {name: None for name in preorder}
    sample_frames = set((0, frame_count // 2, frame_count - 1))
    sampled_motion_error = 0.0
    temp_path = output_path + ".tmp"
    if os.path.isfile(temp_path):
        os.remove(temp_path)

    try:
        with io.open(temp_path, "w", encoding="ascii", newline="\n") as handle:
            handle.write("HIERARCHY\n")
            _write_bvh_joint(
                handle,
                root_name,
                snapshot["children"],
                snapshot["offsets"],
                0,
                True,
            )
            handle.write("MOTION\n")
            handle.write("Frames:\t{}\n".format(frame_count))
            handle.write("Frame Time:\t{:.10f}\n".format(frame_time))

            for frame_index in range(frame_count):
                sample_time = sdk.FBTime()
                sample_time.SetSecondDouble(
                    start_seconds + frame_index * frame_time
                )
                player.Goto(sample_time)
                scene.Evaluate()

                root_model = animated_models[root_name]
                root_delta = _matrix_mult(
                    _global_rotation_matrix(root_model),
                    snapshot["rest_rotation_inverses"][root_name],
                )
                # Blender's BVH importer interprets a joint's position
                # channels as an absolute BVH-space position and subtracts
                # that joint's rest OFFSET when building pose-bone location.
                # Keep the non-zero Root_M rest OFFSET for the correct rest
                # skeleton, but emit Root_M's absolute animated position here.
                root_channel_position = _global_position(root_model)

                bvh_global = {root_name: root_delta}
                values = list(root_channel_position)
                root_euler = _unwrap_euler(
                    _matrix_to_bvh_euler(root_delta),
                    previous_eulers[root_name],
                )
                previous_eulers[root_name] = list(root_euler)
                values.extend(root_euler)

                for name in preorder[1:]:
                    parent = snapshot["parents"][name]
                    parent_global = bvh_global[parent]
                    if animated_rotation[name]:
                        desired_global = _matrix_mult(
                            _global_rotation_matrix(animated_models[name]),
                            snapshot["rest_rotation_inverses"][name],
                        )
                    else:
                        # Static face/hair/accessory joints inherit their
                        # nearest animated parent's rest-space delta.
                        desired_global = parent_global
                    bvh_global[name] = desired_global
                    local_matrix = _matrix_mult(
                        _matrix_inverse(parent_global), desired_global
                    )
                    local_euler = _unwrap_euler(
                        _matrix_to_bvh_euler(local_matrix),
                        previous_eulers[name],
                    )
                    previous_eulers[name] = list(local_euler)
                    values.extend(local_euler)

                if frame_index in sample_frames:
                    reconstructed = {
                        root_name: root_channel_position
                    }
                    sampled_motion_error = max(
                        sampled_motion_error,
                        _distance(
                            reconstructed[root_name],
                            _global_position(animated_models[root_name]),
                        ),
                    )
                    for name in preorder[1:]:
                        parent = snapshot["parents"][name]
                        reconstructed[name] = _vector_add(
                            reconstructed[parent],
                            _matrix_rotate_vector(
                                bvh_global[parent], snapshot["offsets"][name]
                            ),
                        )
                        sampled_motion_error = max(
                            sampled_motion_error,
                            _distance(
                                reconstructed[name],
                                _global_position(animated_models[name]),
                            ),
                        )

                handle.write(
                    " ".join(_bvh_number(value) for value in values) + "\n"
                )
                if frame_index % 600 == 0:
                    print(
                        "[AX retarget] BVH frame {}/{}".format(
                            frame_index, frame_count - 1
                        )
                    )

        if sampled_motion_error > 0.001:
            raise RuntimeError(
                "Rebased BVH sampled FK error is too large: {}".format(
                    sampled_motion_error
                )
            )
        os.replace(temp_path, output_path)
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)

    frames, actual_frame_time, joint_count = _validate_bvh(
        output_path,
        root_name,
        frame_count,
        frame_time,
        len(preorder),
    )
    return {
        "path": output_path,
        "frames": frames,
        "frame_time": actual_frame_time,
        "joint_count": joint_count,
        "animated_rotation_joints": sum(
            1 for value in animated_rotation.values() if value
        ),
        "excluded_scale_bones": snapshot["excluded_scale_bones"],
        "excluded_global_bone": snapshot["excluded_global_bone"],
        "root_position_channels": "absolute_world_position_for_blender",
        "rest_pose_max_error": rest_error,
        "sampled_motion_fk_max_error": sampled_motion_error,
    }


def _export_target_fbx(app, system, take, target_models, output_path):
    # The source skeleton remains in the temporary working scene, but only the
    # models captured immediately after opening the target are selected/saved.
    for model in _all_scene_models(system):
        try:
            model.Selected = False
        except Exception:
            pass
    for model in target_models:
        model.Selected = True

    options = sdk.FBFbxOptions(False)
    for index, candidate_take in enumerate(system.Scene.Takes):
        options.SetTakeSelect(index, candidate_take == take)
    options.SaveSelectedModelsOnly = True
    options.KeepTransformHierarchy = True
    options.Characters = sdk.FBElementAction.kFBElementActionDiscard
    options.CharactersAnimation = False
    options.EmbedMedia = False

    if not app.FileSave(output_path, options):
        raise RuntimeError("MotionBuilder FBX save returned False")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Expected FBX was not created: " + output_path)
    return output_path


def _remove_existing_pair(fbx_path, bvh_path):
    for path in (fbx_path, bvh_path):
        if os.path.isfile(path):
            os.remove(path)


def _process_one(
    app,
    system,
    target_fbx,
    source_fbx,
    fbx_dir,
    bvh_dir,
    overwrite,
    source_type,
):
    base = os.path.splitext(os.path.basename(source_fbx))[0] + OUTPUT_SUFFIX
    fbx_path = os.path.join(fbx_dir, base + ".fbx")
    bvh_path = os.path.join(bvh_dir, base + ".bvh")

    if os.path.exists(fbx_path) or os.path.exists(bvh_path):
        if not overwrite:
            return {
                "status": "skipped",
                "source": source_fbx,
                "source_type": source_type,
                "fbx": fbx_path,
                "bvh": bvh_path,
                "reason": "At least one output already exists; pass --overwrite to replace the pair.",
            }
        _remove_existing_pair(fbx_path, bvh_path)

    # Open target without a namespace so final exported joint names are clean.
    target_options = sdk.FBFbxOptions(True)
    if not app.FileOpen(target_fbx, False, target_options):
        raise RuntimeError("Could not open target FBX: " + target_fbx)
    app.FlushEventQueue()

    target_models = _all_scene_models(system)
    if not target_models:
        raise RuntimeError("The target FBX contains no models")
    target_animation_cleanup = _clear_target_skeleton_animation(system)
    # This snapshot must follow target-animation cleanup, but precede source
    # append and the temporary target T-pose.
    bvh_rest_snapshot = _capture_bvh_rest_skeleton(system)

    source_options = sdk.FBFbxOptions(True)
    source_options.NamespaceList = SOURCE_NAMESPACE
    if not app.FileAppend(source_fbx, False, source_options):
        raise RuntimeError("Could not append source FBX: " + source_fbx)
    app.FlushEventQueue()

    scene = system.Scene
    take = system.CurrentTake
    if take is None:
        raise RuntimeError("The merged scene has no current animation take")

    player = sdk.FBPlayerControl()
    source_time_mode = player.GetTransportFps()
    source_fps = float(player.GetTransportFpsValue())
    take_start = take.LocalTimeSpan.GetStart()
    take_stop = take.LocalTimeSpan.GetStop()

    source_models = [
        model
        for model in scene.ModelSkeletons
        if _long_name(model).startswith(SOURCE_NAMESPACE + ":")
    ]
    if not source_models:
        raise RuntimeError("No source skeleton was found after append")

    _create_source_reference(scene)
    _make_characterization_take(system, take, source_models, source_type)
    source_t_pose_metrics = _validate_source_t_pose(
        scene, SOURCE_TPOSE_VERTICAL_TOLERANCE[source_type]
    )
    target_t_pose_metrics = _validate_target_t_pose(scene)

    source_character = _characterize(
        scene,
        source_type.title() + "_Source",
        SOURCE_MAP,
        SOURCE_NAMESPACE,
    )
    target_character = _characterize(scene, "AX_Female_Target", TARGET_MAP)

    system.CurrentTake = take
    player.SetTransportFps(source_time_mode)
    player.Goto(take_start)
    scene.Evaluate()

    target_character.InputCharacter = source_character
    target_character.InputType = (
        sdk.FBCharacterInputType.kFBCharacterInputCharacter
    )
    target_character.ActiveInput = True
    scene.Evaluate()

    # PlotAnimation plus post-plot curve checks are used as the authoritative
    # success criteria for this programmatically constructed character pair.
    if not target_character.PlotAnimation(
        sdk.FBCharacterPlotWhere.kFBCharacterPlotOnSkeleton,
        _make_plot_options(),
    ):
        raise RuntimeError("MotionBuilder failed to plot retargeted animation")

    target_character.ActiveInput = False
    system.CurrentTake = take
    player.SetTransportFps(source_time_mode)
    player.Goto(take_start)
    scene.Evaluate()
    thumb_root_correction = None
    if source_type == "zeroeggs":
        thumb_root_correction = _apply_thumb_root_rebased_correction(
            system,
            take,
            bvh_rest_snapshot,
            source_fps,
        )
        system.CurrentTake = take
        player.SetTransportFps(source_time_mode)
        player.Goto(take_start)
        scene.Evaluate()
    baked_key_counts = _validate_baked_animation(scene)
    retargeted_arm_samples = _sample_retargeted_arms(
        system, take, source_fps
    )

    take_name = _safe_text(take.Name)
    take_start_seconds = float(take_start.GetSecondDouble())
    take_stop_seconds = float(take_stop.GetSecondDouble())
    target_model_count = len(target_models)

    bvh_result = _export_bvh_rebased(
        system,
        take,
        bvh_rest_snapshot,
        bvh_path,
        source_fps,
    )
    fbx_path = _export_target_fbx(
        app, system, take, target_models, fbx_path
    )

    return {
        "status": "converted",
        "source": source_fbx,
        "source_type": source_type,
        "source_t_pose_vertical_tolerance": (
            SOURCE_TPOSE_VERTICAL_TOLERANCE[source_type]
        ),
        "fbx": fbx_path,
        "bvh": bvh_path,
        "source_fps": source_fps,
        "take": take_name,
        "take_start_seconds": take_start_seconds,
        "take_stop_seconds": take_stop_seconds,
        "bvh_frames": bvh_result["frames"],
        "bvh_frame_time": bvh_result["frame_time"],
        "bvh_root": "Root_M",
        "bvh_kept_joints": bvh_result["joint_count"],
        "bvh_animated_rotation_joints": bvh_result[
            "animated_rotation_joints"
        ],
        "bvh_removed_scale_joint_count": len(
            bvh_result["excluded_scale_bones"]
        ),
        "bvh_removed_scale_joints": bvh_result["excluded_scale_bones"],
        "bvh_removed_global_bone": bvh_result["excluded_global_bone"],
        "bvh_root_position_channels": bvh_result[
            "root_position_channels"
        ],
        "bvh_rest_pose_max_error": bvh_result["rest_pose_max_error"],
        "bvh_sampled_motion_fk_max_error": bvh_result[
            "sampled_motion_fk_max_error"
        ],
        "fbx_target_models": target_model_count,
        "target_animation_cleanup": target_animation_cleanup,
        "baked_key_counts": baked_key_counts,
        "thumb_root_correction": thumb_root_correction,
        "source_t_pose_positions": source_t_pose_metrics,
        "target_t_pose_positions": target_t_pose_metrics,
        "retargeted_arm_samples": retargeted_arm_samples,
        "fbx_bytes": os.path.getsize(fbx_path),
        "bvh_bytes": os.path.getsize(bvh_result["path"]),
    }


def _collect_sources(source_dir, source_file, max_files):
    if source_file:
        sources = [os.path.abspath(source_file)]
    else:
        sources = []
        for name in os.listdir(source_dir):
            full_path = os.path.join(source_dir, name)
            if os.path.isfile(full_path) and name.lower().endswith(".fbx"):
                sources.append(os.path.abspath(full_path))
        sources.sort(key=lambda value: value.lower())
    if max_files > 0:
        sources = sources[:max_files]
    return sources


def _write_report(path, report):
    with io.open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write(u"\n")


def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Retarget ZeroEGGS or Motorica FBX animation to AX_female2 and export "
            "target FBX plus Blender-friendly BVH."
        )
    )
    parser.add_argument(
        "--target-fbx",
        required=True,
        help="Target AX_female2 FBX path.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-dir",
        help="Directory of source FBX files to process in sorted order.",
    )
    source_group.add_argument(
        "--source-file",
        help="Process one source FBX instead of a directory.",
    )
    parser.add_argument(
        "--source-type",
        choices=SOURCE_TYPES,
        required=True,
        help="Source FBX layout: zeroeggs or motorica.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root; fbx/, bvh/, and retarget_report.json are created here.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum sorted source files to process; 0 means all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing same-name output FBX/BVH pair. "
            "Without this flag, that source is skipped."
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split the sorted source list into this many interleaved workers.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based worker index used with --shard-count.",
    )
    parser.add_argument(
        "--report-name",
        default="retarget_report.json",
        help="Report filename inside --output-dir (use a unique name per worker).",
    )
    args = parser.parse_args(argv)
    if args.max_files < 0:
        parser.error("--max-files must be zero or a positive integer")
    if args.shard_count < 1:
        parser.error("--shard-count must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if os.path.basename(args.report_name) != args.report_name:
        parser.error("--report-name must be a filename, not a path")
    return args


def main():
    app = sdk.FBApplication()
    system = sdk.FBSystem()
    args = _parse_cli_args()

    target_fbx = os.path.abspath(args.target_fbx)
    source_dir = (
        os.path.abspath(args.source_dir) if args.source_dir else ""
    )
    source_file = (
        os.path.abspath(args.source_file) if args.source_file else ""
    )
    output_root = os.path.abspath(args.output_dir)
    source_type = args.source_type
    max_files = args.max_files
    overwrite = args.overwrite

    if not os.path.isfile(target_fbx):
        raise RuntimeError("Target FBX does not exist: " + target_fbx)
    if source_file and not os.path.isfile(source_file):
        raise RuntimeError("--source-file does not exist: " + source_file)
    if not source_file and not os.path.isdir(source_dir):
        raise RuntimeError("Source directory does not exist: " + source_dir)

    all_sources = _collect_sources(source_dir, source_file, max_files)
    sources = all_sources[args.shard_index :: args.shard_count]
    if not sources:
        raise RuntimeError("No source FBX files were assigned to this shard")

    fbx_dir = os.path.join(output_root, "fbx")
    bvh_dir = os.path.join(output_root, "bvh")
    _make_dir(fbx_dir)
    _make_dir(bvh_dir)

    started = datetime.datetime.now().isoformat()
    report = {
        "status": "running",
        "started_at": started,
        "target_fbx": target_fbx,
        "source_dir": source_dir or None,
        "source_file_override": source_file or None,
        "source_type": source_type,
        "output_root": output_root,
        "overwrite": overwrite,
        "discovered_files": len(all_sources),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "requested_files": len(sources),
        "files": [],
    }
    report_path = os.path.join(output_root, args.report_name)
    _write_report(report_path, report)

    counts = {"converted": 0, "skipped": 0, "failed": 0}
    for index, source_fbx in enumerate(sources, 1):
        print(
            "[AX retarget] {}/{} {}".format(
                index, len(sources), source_fbx
            )
        )
        try:
            result = _process_one(
                app,
                system,
                target_fbx,
                source_fbx,
                fbx_dir,
                bvh_dir,
                overwrite,
                source_type,
            )
            counts[result["status"]] += 1
            report["files"].append(result)
            print(
                "[AX retarget] {} {}".format(
                    result["status"].upper(), source_fbx
                )
            )
        except Exception as exc:
            counts["failed"] += 1
            result = {
                "status": "failed",
                "source": source_fbx,
                "error": _safe_text(exc),
                "traceback": traceback.format_exc(),
            }
            report["files"].append(result)
            print("[AX retarget] FAILED: " + _safe_text(exc))
            traceback.print_exc()
        finally:
            report["counts"] = counts.copy()
            _write_report(report_path, report)
            try:
                app.FileNew(False)
            except Exception:
                pass

    report["status"] = "finished"
    report["finished_at"] = datetime.datetime.now().isoformat()
    report["counts"] = counts
    _write_report(report_path, report)

    print(
        "[AX retarget] Finished | converted={} skipped={} failed={} | report={}".format(
            counts["converted"],
            counts["skipped"],
            counts["failed"],
            report_path,
        )
    )
    return 0 if counts["failed"] == 0 else 1


if __name__ in ("__main__", "builtins", "__builtin__"):
    try:
        _exit_code = main()
    except Exception:
        traceback.print_exc()
        _exit_code = 2
    if _is_noninteractive(sdk.FBApplication()):
        raise SystemExit(_exit_code)
