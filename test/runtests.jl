using Test
using MACEInterface
using PythonCall

@testset "MACEInterface.jl" begin

    @testset "Imports" begin
        @test !isnothing(pyimport("numpy"))
        @test !isnothing(pyimport("ase.atoms"))
        @test !isnothing(pyimport("mace.calculators"))
    end

    @testset "Unit system" begin
        u = unit_system()
        @test u.energy == "eV"
        @test u.length == "Angstrom"
        @test u.force == "eV/Angstrom"
        @test u.stress == "eV/Angstrom^3"
        @test u.virial == "eV"

        @test energy_unit() == "eV"
        @test force_unit() == "eV/Angstrom"
        @test stress_unit() == "eV/Angstrom^3"
        @test virial_unit() == "eV"
    end

    @testset "Constructor validation" begin
        @test_throws ArgumentError MACEPotential(
            "no_such_file.model",
            ["H"],
            zeros(1, 3),
        )

        model_path = joinpath(@__DIR__, "2023-12-10-mace-128-L0_energy_epoch-249.model")

        if isfile(model_path)
            symbols = ["O", "H", "H"]
            x0 = [
                0.000000 0.000000 0.000000
                0.758602 0.000000 0.504284
                -0.758602 0.000000 0.504284
            ]

            pot = MACEPotential(model_path, symbols, x0; device="cpu")

            @test_throws ArgumentError set_positions!(pot, zeros(2, 3))
            @test_throws ArgumentError set_positions!(pot, zeros(3, 2))
            @test_throws ArgumentError set_cell!(pot, zeros(2, 2))
            @test_throws ArgumentError set_cell!(pot, zeros(3, 2))
        else
            @info "Skipping shape-validation tests requiring a real model."
        end
    end

    model_path = joinpath(@__DIR__, "2023-12-10-mace-128-L0_energy_epoch-249.model")

    if isfile(model_path)

        @testset "Integration with test model" begin
            symbols = ["O", "H", "H"]

            x0 = [
                0.000000 0.000000 0.000000
                0.758602 0.000000 0.504284
                -0.758602 0.000000 0.504284
            ]

            cell0 = [
                10.0 0.0 0.0
                0.0 10.0 0.0
                0.0 0.0 10.0
            ]

            pot = MACEPotential(
                model_path,
                symbols,
                x0;
                cell=cell0,
                pbc=(true, true, true),
                device="cpu",
            )

            @test natoms(pot) == 3

            c0 = cell(pot)
            @test size(c0) == (3, 3)
            @test isapprox(c0, cell0; atol=1e-12)

            @test pbc(pot) == (true, true, true)

            V0 = volume(pot)
            @test V0 isa Float64
            @test isfinite(V0)
            @test isapprox(V0, 1000.0; atol=1e-10)

            e0 = energy(pot)
            f0 = forces(pot)

            @test e0 isa Float64
            @test isfinite(e0)

            @test size(f0) == (3, 3)
            @test eltype(f0) == Float64
            @test all(isfinite, f0)

            s0 = stress(pot)
            @test size(s0) == (3, 3)
            @test eltype(s0) == Float64
            @test all(isfinite, s0)

            s0v = stress(pot; voigt=true)
            @test length(s0v) == 6
            @test eltype(s0v) == Float64
            @test all(isfinite, s0v)

            w0 = virial(pot)
            @test size(w0) == (3, 3)
            @test eltype(w0) == Float64
            @test all(isfinite, w0)

            @test isapprox(w0, -V0 .* s0; atol=1e-10)

            @testset "Position updates" begin
                x1 = copy(x0)
                x1[2, 3] += 0.01

                set_positions!(pot, x1)

                e1 = energy(pot)
                f1 = forces(pot)

                @test e1 isa Float64
                @test size(f1) == (3, 3)
                @test all(isfinite, f1)

                @test !(e0 == e1 && f0 == f1)
            end

            @testset "Combined evaluation APIs" begin
                x2 = copy(x0)
                x2[1, 1] += 0.02

                e2, f2 = energy_forces(pot, x2)
                @test e2 isa Float64
                @test size(f2) == (3, 3)
                @test all(isfinite, f2)

                e3, f3, s3 = energy_forces_stress(pot, x2)
                @test e3 isa Float64
                @test size(f3) == (3, 3)
                @test size(s3) == (3, 3)
                @test all(isfinite, f3)
                @test all(isfinite, s3)

                e4, f4, w4 = energy_forces_virial(pot, x2)
                @test e4 isa Float64
                @test size(f4) == (3, 3)
                @test size(w4) == (3, 3)
                @test all(isfinite, f4)
                @test all(isfinite, w4)

                @test isapprox(w4, -volume(pot) .* stress(pot); atol=1e-10)
            end

            @testset "Cell updates" begin
                newcell = [
                    11.0 0.0 0.0
                    0.0 10.0 0.0
                    0.0 0.0 10.0
                ]

                set_cell!(pot, newcell; scale_atoms=false)

                c1 = cell(pot)
                @test size(c1) == (3, 3)
                @test isapprox(c1, newcell; atol=1e-12)

                V1 = volume(pot)
                @test isfinite(V1)
                @test isapprox(V1, 1100.0; atol=1e-10)
            end

            @testset "PBC updates" begin
                set_pbc!(pot, (true, false, true))
                @test pbc(pot) == (true, false, true)

                set_pbc!(pot, (false, false, false))
                @test pbc(pot) == (false, false, false)

                set_pbc!(pot, (true, true, true))
                @test pbc(pot) == (true, true, true)
            end
        end

    else
        @info "Skipping integration test; model not found at $model_path"
    end
end