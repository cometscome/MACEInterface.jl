module MACEInterface

using PythonCall

export MACEPotential,
    set_positions!,
    set_cell!,
    set_pbc!,
    energy,
    forces,
    stress,
    virial,
    volume,
    natoms,
    cell,
    pbc,
    energy_forces,
    energy_forces_stress,
    energy_forces_virial,
    unit_system,
    energy_unit,
    force_unit,
    stress_unit,
    virial_unit

const _ase_atoms = Ref{Py}()
const _mace_calcs = Ref{Py}()
const _np = Ref{Py}()

function __init__()
    _ase_atoms[] = pyimport("ase.atoms")
    _mace_calcs[] = pyimport("mace.calculators")
    _np[] = pyimport("numpy")
end

mutable struct MACEPotential
    calc::Py
    atoms::Py
    natoms::Int
end

"""
    MACEPotential(model_path, symbols, positions; cell=nothing, pbc=nothing,
                  device="cpu", default_dtype="float64")

Create a MACE calculator and an ASE Atoms object, and keep them alive
for repeated evaluations in the current Julia session.

Arguments
- `model_path`: path to a trained MACE model file
- `symbols`: vector of chemical symbols
- `positions`: `natoms × 3` matrix

Keyword arguments
- `cell`: optional `3 × 3` cell matrix
- `pbc`: optional periodic boundary condition flags, e.g. `(true, true, true)`
- `device`: `"cpu"` or `"cuda"`
- `default_dtype`: usually `"float64"` or `"float32"`
"""
function MACEPotential(
    model_path::AbstractString,
    symbols::Vector{String},
    positions::AbstractMatrix{<:Real};
    cell::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    pbc::Union{Nothing,NTuple{3,Bool}}=nothing,
    device::String="cpu",
    default_dtype::String="float64",
)
    isfile(model_path) || throw(ArgumentError("model file not found: $model_path"))

    natoms_ = length(symbols)
    _check_positions(positions, natoms_)

    calc = _mace_calcs[].MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype=default_dtype,
    )

    kwargs = Dict{Symbol,Any}()
    kwargs[:symbols] = symbols
    kwargs[:positions] = _to_numpy_matrix(positions)

    if cell !== nothing
        _check_cell(cell)
        kwargs[:cell] = _to_numpy_matrix(cell)
    end

    if pbc !== nothing
        kwargs[:pbc] = collect(pbc)
    end

    atoms = _ase_atoms[].Atoms(; kwargs...)
    atoms.calc = calc

    return MACEPotential(calc, atoms, natoms_)
end

# ----------------------------
# Internal helpers
# ----------------------------

function _to_numpy_matrix(x::AbstractMatrix{<:Real})
    return _np[].array(Matrix{Float64}(x), dtype=_np[].float64)
end

function _check_positions(positions::AbstractMatrix, natoms_::Integer)
    size(positions, 1) == natoms_ ||
        throw(ArgumentError("positions must have size ($(natoms_), 3), got $(size(positions))"))
    size(positions, 2) == 3 ||
        throw(ArgumentError("positions must have size ($(natoms_), 3), got $(size(positions))"))
    return nothing
end

function _check_cell(cell::AbstractMatrix)
    size(cell, 1) == 3 && size(cell, 2) == 3 ||
        throw(ArgumentError("cell must have size (3, 3), got $(size(cell))"))
    return nothing
end

function _check_pbc_flags(pbc::NTuple{3,Bool})
    return nothing
end

# ----------------------------
# Structure mutators
# ----------------------------

"""
    set_positions!(pot, positions)

Update atomic positions in the stored ASE Atoms object.
"""
function set_positions!(pot::MACEPotential, positions::AbstractMatrix{<:Real})
    _check_positions(positions, pot.natoms)
    pot.atoms.positions = _to_numpy_matrix(positions)
    return pot
end

"""
    set_cell!(pot, cell; scale_atoms=false)

Update the simulation cell.

If `scale_atoms=true`, ASE rescales atom positions together with the cell.
"""
function set_cell!(
    pot::MACEPotential,
    newcell::AbstractMatrix{<:Real};
    scale_atoms::Bool=false,
)
    _check_cell(newcell)
    pot.atoms.set_cell(_to_numpy_matrix(newcell), scale_atoms=scale_atoms)
    return pot
end

"""
    set_pbc!(pot, pbc)

Update periodic boundary condition flags.
Example: `(true, true, true)`.
"""
function set_pbc!(pot::MACEPotential, newpbc::NTuple{3,Bool})
    _check_pbc_flags(newpbc)
    pot.atoms.pbc = collect(newpbc)
    return pot
end

# ----------------------------
# Structure accessors
# ----------------------------

"""
    natoms(pot)

Number of atoms.
"""
natoms(pot::MACEPotential) = pot.natoms

"""
    cell(pot)

Return the current cell as a `3 × 3` Julia matrix.
"""
function cell(pot::MACEPotential)
    return pyconvert(Matrix{Float64}, pot.atoms.cell.array)
end

"""
    pbc(pot)

Return the periodic boundary condition flags as a 3-tuple of Bool.
"""
function pbc(pot::MACEPotential)
    flags = pyconvert(Vector{Bool}, pot.atoms.pbc)
    length(flags) == 3 || throw(ErrorException("unexpected pbc length: $(length(flags))"))
    return (flags[1], flags[2], flags[3])
end

"""
    volume(pot)

Return the cell volume.

This requires that a valid cell is defined in the ASE Atoms object.
"""
function volume(pot::MACEPotential)
    return pyconvert(Float64, pot.atoms.get_volume())
end

# ----------------------------
# Energy / forces / stress / virial
# ----------------------------

"""
    energy(pot)

Potential energy.
"""
function energy(pot::MACEPotential)
    return pyconvert(Float64, pot.atoms.get_potential_energy())
end

"""
    forces(pot)

Atomic forces as a `natoms × 3` Julia matrix.
"""
function forces(pot::MACEPotential)
    return pyconvert(Matrix{Float64}, pot.atoms.get_forces())
end

"""
    stress(pot; voigt=false)

Configuration stress.

- `voigt=false`: returns a `3 × 3` matrix
- `voigt=true`: returns a 6-component Voigt vector
"""
function stress(pot::MACEPotential; voigt::Bool=false)
    s = pot.atoms.get_stress(voigt=voigt)
    if voigt
        return pyconvert(Vector{Float64}, s)
    else
        return pyconvert(Matrix{Float64}, s)
    end
end

"""
    virial(pot)

Configuration virial tensor as a `3 × 3` matrix.

Computed from the stress tensor by

    W = -V * σ

where `V` is the cell volume and `σ` is the configuration stress.
"""
function virial(pot::MACEPotential)
    σ = stress(pot; voigt=false)
    V = volume(pot)
    return -V .* σ
end

# ----------------------------
# Combined evaluation helpers
# ----------------------------

"""
    energy_forces(pot, positions)

Update positions and return `(energy, forces)`.
"""
function energy_forces(pot::MACEPotential, positions::AbstractMatrix{<:Real})
    set_positions!(pot, positions)
    return energy(pot), forces(pot)
end

"""
    energy_forces_stress(pot, positions)

Update positions and return `(energy, forces, stress)`.
"""
function energy_forces_stress(pot::MACEPotential, positions::AbstractMatrix{<:Real})
    set_positions!(pot, positions)
    return energy(pot), forces(pot), stress(pot; voigt=false)
end

"""
    energy_forces_virial(pot, positions)

Update positions and return `(energy, forces, virial)`.
"""
function energy_forces_virial(pot::MACEPotential, positions::AbstractMatrix{<:Real})
    set_positions!(pot, positions)
    return energy(pot), forces(pot), virial(pot)
end

# ----------------------------
# Unit-system helpers
# ----------------------------

"""
    unit_system()

Return a NamedTuple describing the unit convention used by this interface.

This interface assumes the standard ASE/MACE convention:
- energy: eV
- length: Å
- force: eV/Å
- stress: eV/Å^3
- virial: eV
"""
function unit_system()
    return (
        energy="eV",
        length="Angstrom",
        force="eV/Angstrom",
        stress="eV/Angstrom^3",
        virial="eV",
    )
end

"""
    energy_unit()

Return the energy unit string.
"""
energy_unit() = unit_system().energy

"""
    force_unit()

Return the force unit string.
"""
force_unit() = unit_system().force

"""
    stress_unit()

Return the stress unit string.
"""
stress_unit() = unit_system().stress

"""
    virial_unit()

Return the virial unit string.
"""
virial_unit() = unit_system().virial

end