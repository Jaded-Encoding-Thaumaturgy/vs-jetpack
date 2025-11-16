JETPACK_MODULES = (
    "vsaa",
    "vsdeband",
    "vsdehalo",
    "vsdeinterlace",
    "vsdenoise",
    "vsexprtools",
    "vskernels",
    "vsmasktools",
    "vsrgtools",
    "vsscale",
    "vssource",
    "vstools",
)

JETPACK_MODULES_ALIASES = {
    **{m.removeprefix("vs"): m for m in JETPACK_MODULES},
    "expr": "vsexprtools",
    "mask": "vsmasktools",
    "rg": "vsrgtools",
}
