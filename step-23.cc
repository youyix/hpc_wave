/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2006
 */


// @sect3{Include files}

// We start with the usual assortment of include files that we've seen in so
// many of the previous tests:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/base/timer.h>



#include <fstream>
#include <iostream>

// Here are the only three include files of some new interest: The first one
// is already used, for example, for the
// VectorTools::interpolate_boundary_values and
// MatrixTools::apply_boundary_values functions. However, we here use another
// function in that class, VectorTools::project to compute our initial values
// as the $L^2$ projection of the continuous initial values. Furthermore, we
// use VectorTools::create_right_hand_side to generate the integrals
// $(f^n,\phi^n_i)$. These were previously always generated by hand in
// <code>assemble_system</code> or similar functions in application
// code. However, we're too lazy to do that here, so simply use a library
// function:
#include <deal.II/numerics/vector_tools.h>

// In a very similar vein, we are also too lazy to write the code to assemble
// mass and Laplace matrices, although it would have only taken copying the
// relevant code from any number of previous tutorial programs. Rather, we
// want to focus on the things that are truly new to this program and
// therefore use the MatrixCreator::create_mass_matrix and
// MatrixCreator::create_laplace_matrix functions. They are declared here:
#include <deal.II/numerics/matrix_tools.h>

// Finally, here is an include file that contains all sorts of tool functions
// that one sometimes needs. In particular, we need the
// Utilities::int_to_string class that, given an integer argument, returns a
// string representation of it. It is particularly useful since it allows for
// a second parameter indicating the number of digits to which we want the
// result padded with leading zeros. We will use this to write output files
// that have the form <code>solution-XXX.gnuplot</code> where <code>XXX</code>
// denotes the number of the time step and always consists of three digits
// even if we are still in the single or double digit time steps.
#include <deal.II/base/utilities.h>

// The last step is as in all previous programs:
namespace Step23
{
  using namespace dealii;

    // @sect3{Equation data}

  // Before we go on filling in the details of the main class, let us define
  // the equation data corresponding to the problem, i.e. initial and boundary
  // values for both the solution $u$ and its time derivative $v$, as well as
  // a right hand side class. We do so using classes derived from the Function
  // class template that has been used many times before, so the following
  // should not be a surprise.
  //
  // Let's start with initial values and choose zero for both the value $u$ as
  // well as its time derivative, the velocity $v$:
  template <int dim>
  class InitialValuesU : public Function<dim>
  {
  public:
    InitialValuesU () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  class InitialValuesV : public Function<dim>
  {
  public:
    InitialValuesV () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };



  template <int dim>
  double InitialValuesU<dim>::value (const Point<dim>  &p,
                                     const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());
    return 0;
  }



  template <int dim>
  double InitialValuesV<dim>::value (const Point<dim>  &p,
                                     const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());
    return 0;
  }



  // Secondly, we have the right hand side forcing term. Boring as we are, we
  // choose zero here as well:
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                    std::vector<Vector<double> >   &value_list) const;
  };



  template <int dim>
  double RightHandSide<dim>::value (const Point<dim>  &p,
                                    const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());
    return 0;
  }

  template <int dim>
  void RightHandSide<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                              std::vector<Vector<double> >   &value_list) const
  {
    const unsigned int n_points = points.size();

    Assert (value_list.size() == n_points,
            ExcDimensionMismatch (value_list.size(), n_points));

    for (unsigned int p=0; p<n_points; ++p)
      RightHandSide<dim>::vector_value (points[p],
                                        value_list[p]);
  }



  // Finally, we have boundary values for $u$ and $v$. They are as described
  // in the introduction, one being the time derivative of the other:
  template <int dim>
  class BoundaryValuesU : public Function<dim>
  {
  public:
    BoundaryValuesU () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };




  template <int dim>
  class BoundaryValuesV : public Function<dim>
  {
  public:
    BoundaryValuesV () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };




  template <int dim>
  double BoundaryValuesU<dim>::value (const Point<dim> &p,
                                      const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());

    if ((this->get_time() <= 0.5) &&
        (p[0] < 0) &&
        (p[1] < 1./3) &&
        (p[1] > -1./3))
      return std::sin (this->get_time() * 4 * numbers::PI);
    else
      return 0;
  }



  template <int dim>
  double BoundaryValuesV<dim>::value (const Point<dim> &p,
                                      const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());

    if ((this->get_time() <= 0.5) &&
        (p[0] < 0) &&
        (p[1] < 1./3) &&
        (p[1] > -1./3))
      return (std::cos (this->get_time() * 4 * numbers::PI) *
              4 * numbers::PI);
    else
      return 0;
  }


  // @sect3{The <code>WaveEquation</code> class}

  // Next comes the declaration of the main class. It's public interface of
  // functions is like in most of the other tutorial programs. Worth
  // mentioning is that we now have to store four matrices instead of one: the
  // mass matrix $M$, the Laplace matrix $A$, the matrix $M+k^2\theta^2A$ used
  // for solving for $U^n$, and a copy of the mass matrix with boundary
  // conditions applied used for solving for $V^n$. Note that it is a bit
  // wasteful to have an additional copy of the mass matrix around. We will
  // discuss strategies for how to avoid this in the section on possible
  // improvements.
  //
  // Likewise, we need solution vectors for $U^n,V^n$ as well as for the
  // corresponding vectors at the previous time step, $U^{n-1},V^{n-1}$. The
  // <code>system_rhs</code> will be used for whatever right hand side vector
  // we have when solving one of the two linear systems in each time
  // step. These will be solved in the two functions <code>solve_u</code> and
  // <code>solve_v</code>.
  //
  // Finally, the variable <code>theta</code> is used to indicate the
  // parameter $\theta$ that is used to define which time stepping scheme to
  // use, as explained in the introduction. The rest is self-explanatory.
  template <int dim>
  class WaveEquation
  {
  public:
    WaveEquation ();
    void run ();
    void test ();

  private:
    void setup_system ();
    void solve_u ();
    void solve_v ();
    void output_results () const;
    void assemble_mass_matrix ();
    void assemble_laplace_matrix ();
    void hehe ();
    

    Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;

    // ConstraintMatrix hanging_node_constraints;
    ConstraintMatrix     hanging_node_constraints;

    PETScWrappers::MPI::SparseMatrix mass_matrix;
    PETScWrappers::MPI::SparseMatrix laplace_matrix;
    PETScWrappers::MPI::SparseMatrix matrix_u;
    PETScWrappers::MPI::SparseMatrix matrix_v;

    PETScWrappers::MPI::Vector       solution_u, solution_v;
    PETScWrappers::MPI::Vector       old_solution_u, old_solution_v;
    PETScWrappers::MPI::Vector       system_rhs;

    double time, time_step;
    unsigned int timestep_number;
    const double theta;
    MPI_Comm mpi_communicator;

    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    ConditionalOStream                        pcout;

    TimerOutput                               computing_timer;
    TimerOutput                               computing_timer_wall;
  };








  // @sect3{Implementation of the <code>WaveEquation</code> class}

  // The implementation of the actual logic is actually fairly short, since we
  // relegate things like assembling the matrices and right hand side vectors
  // to the library. The rest boils down to not much more than 130 lines of
  // actual code, a significant fraction of which is boilerplate code that can
  // be taken from previous example programs (e.g. the functions that solve
  // linear systems, or that generate output).
  //
  // Let's start with the constructor (for an explanation of the choice of
  // time step, see the section on Courant, Friedrichs, and Lewy in the
  // introduction):
  template <int dim>
  WaveEquation<dim>::WaveEquation () :
    fe (1),
    dof_handler (triangulation),
    time_step (1./64),
    theta (0.5),
    mpi_communicator (MPI_COMM_WORLD),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(mpi_communicator)
            == 0)),
    computing_timer (mpi_communicator,
                     pcout,
                     TimerOutput::summary,
                     // TimerOutput::wall_times,
                     TimerOutput::cpu_times),
    computing_timer_wall (mpi_communicator,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::wall_times)

  {
    pcout.set_condition(this_mpi_process == 0);
  }


  // @sect4{WaveEquation::setup_system}

  // The next function is the one that sets up the mesh, DoFHandler, and
  // matrices and vectors at the beginning of the program, i.e. before the
  // first time step. The first few lines are pretty much standard if you've
  // read through the tutorial programs at least up to step-6:
  template <int dim>
  void WaveEquation<dim>::setup_system ()
  {
    TimerOutput::Scope t(computing_timer, "setup");
    TimerOutput::Scope twall(computing_timer_wall, "setup");


    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (5);

    pcout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;

    GridTools::partition_triangulation (n_mpi_processes, triangulation);

    dof_handler.distribute_dofs (fe);
    DoFRenumbering::subdomain_wise (dof_handler);

    pcout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    const types::global_dof_index n_local_dofs
      = DoFTools::count_dofs_with_subdomain_association (dof_handler,
                                                         this_mpi_process);

    mass_matrix.reinit (mpi_communicator,
                          dof_handler.n_dofs(),
                          dof_handler.n_dofs(),
                          n_local_dofs,
                          n_local_dofs,
                          dof_handler.max_couplings_between_dofs());

    laplace_matrix.reinit (mpi_communicator,
                          dof_handler.n_dofs(),
                          dof_handler.n_dofs(),
                          n_local_dofs,
                          n_local_dofs,
                          dof_handler.max_couplings_between_dofs());

    matrix_u.reinit (mpi_communicator,
                          dof_handler.n_dofs(),
                          dof_handler.n_dofs(),
                          n_local_dofs,
                          n_local_dofs,
                          dof_handler.max_couplings_between_dofs());

    matrix_v.reinit (mpi_communicator,
                          dof_handler.n_dofs(),
                          dof_handler.n_dofs(),
                          n_local_dofs,
                          n_local_dofs,
                          dof_handler.max_couplings_between_dofs());


    assemble_mass_matrix ();
    assemble_laplace_matrix ();
    
    solution_u.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
    solution_v.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
    old_solution_u.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
    old_solution_v.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
    system_rhs.reinit (mpi_communicator, dof_handler.n_dofs(), n_local_dofs);

    // DoFTools::make_hanging_node_constraints
    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             hanging_node_constraints);
    hanging_node_constraints.close ();
  }


  

  
  template <int dim>
  void WaveEquation<dim>::solve_u ()
  {
    TimerOutput::Scope t(computing_timer, "solve_u");
    TimerOutput::Scope twall(computing_timer_wall, "solve_u");
    SolverControl           solver_control (1000, 1e-8*system_rhs.l2_norm());
    // SolverCG<>              cg (solver_control);

    // // SolverControl cn;
    // // PETScWrappers::SparseDirectMUMPS solver(cn, mpi_communicator);
    // // // solver.set_symmetric_mode(true);
    // // solver.solve(matrix_u, solution_u, system_rhs);

    // cg.solve (matrix_u, solution_u, system_rhs,
    //           PreconditionIdentity());

    // std::cout << "   u-equation: " << solver_control.last_step()
    //           << " CG iterations."
    //           << std::endl;
    //           
    
    PETScWrappers::SolverCG cg (solver_control,
                                mpi_communicator);

    PETScWrappers::PreconditionNone preconditioner(matrix_u);

    cg.solve (matrix_u, solution_u, system_rhs, preconditioner);

    pcout << "   u-equation: " << solver_control.last_step()
              << " CG iterations."
              << std::endl;
  }


  template <int dim>
  void WaveEquation<dim>::solve_v ()
  {
    TimerOutput::Scope t(computing_timer, "solve_v");
    TimerOutput::Scope twall(computing_timer_wall, "solve_v");
    SolverControl           solver_control (1000, 1e-8*system_rhs.l2_norm());
    // SolverCG<>              cg (solver_control);

    // cg.solve (matrix_v, solution_v, system_rhs,
    //           PreconditionIdentity());
    //           
    PETScWrappers::SolverCG cg (solver_control,
                                mpi_communicator);
    //           
    PETScWrappers::PreconditionNone preconditioner(matrix_v);

    cg.solve (matrix_v, solution_v, system_rhs, preconditioner);

    pcout << "   v-equation: " << solver_control.last_step()
              << " CG iterations."
              << std::endl;
  }



  // @sect4{WaveEquation::output_results}

  // Likewise, the following function is pretty much what we've done
  // before. The only thing worth mentioning is how here we generate a string
  // representation of the time step number padded with leading zeros to 3
  // character length using the Utilities::int_to_string function's second
  // argument.
  template <int dim>
  void WaveEquation<dim>::output_results () const
  {
    // Gather data from all processes.
    const PETScWrappers::Vector localized_solution_u (solution_u);
    const PETScWrappers::Vector localized_solution_v (solution_v);

    if ( this_mpi_process == 0 ) {
      DataOut<dim> data_out;

      data_out.attach_dof_handler (dof_handler);
      data_out.add_data_vector (localized_solution_u, "U");
      data_out.add_data_vector (localized_solution_v, "V");

      data_out.build_patches ();

      const std::string filename = "solution-" +
                                   Utilities::int_to_string (timestep_number, 3) +
                                   ".vtk";
      std::ofstream output (filename.c_str());
      data_out.write_vtk (output);
    }
    
  }

  template <int dim>
  void WaveEquation<dim>::test ()
  {

  }

  template <int dim>
  void WaveEquation<dim>::assemble_mass_matrix ()
  {
    TimerOutput::Scope t(computing_timer, "assemble_mass_matrix");
    TimerOutput::Scope twall(computing_timer_wall, "assemble_mass_matrix");
    QGauss<dim>   quadrature_formula(3);
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;

    const unsigned int   n_q_points    = quadrature_formula.size();


    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    
    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

    int koko = 0;
    for (; cell!=endc; ++cell)
    {
      if (cell->subdomain_id() == this_mpi_process)
      {
        koko++;
        fe_values.reinit (cell);
        cell_matrix = 0;

        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j) 
              {
                cell_matrix(i,j) += (fe_values.shape_value (i, q_index) *
                                     fe_values.shape_value (j, q_index) *
                                     fe_values.JxW (q_index));

              }
            }
        
        cell->get_dof_indices (local_dof_indices);
        hanging_node_constraints
        .distribute_local_to_global(cell_matrix, 
                                    local_dof_indices,
                                    mass_matrix); 
      }
      
    }
    mass_matrix.compress(VectorOperation::add);
  }

  template <int dim>
  void WaveEquation<dim>::assemble_laplace_matrix ()
  {
    TimerOutput::Scope t(computing_timer, "assemble_laplace_matrix");
    TimerOutput::Scope twall(computing_timer_wall, "assemble_laplace_matrix");
    QGauss<dim>   quadrature_formula(3);
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;

    const unsigned int   n_q_points    = quadrature_formula.size();


    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    
    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

    int koko = 0;
    for (; cell!=endc; ++cell)
    {
      if (cell->subdomain_id() == this_mpi_process)
      {
        koko++;
        fe_values.reinit (cell);
        cell_matrix = 0;
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j) 
              {
                cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                                     fe_values.shape_grad (j, q_index) *
                                     fe_values.JxW (q_index));

              }
            }

        cell->get_dof_indices (local_dof_indices);
        hanging_node_constraints
        .distribute_local_to_global(cell_matrix, 
                                    local_dof_indices,
                                    laplace_matrix); 
      }
    }
    laplace_matrix.compress(VectorOperation::add);
  }

    template <int dim>
  void WaveEquation<dim>::hehe ()
  {
    TimerOutput::Scope t(computing_timer, "hehe");
    TimerOutput::Scope twall(computing_timer_wall, "hehe");

    const types::global_dof_index n_local_dofs
      = DoFTools::count_dofs_with_subdomain_association (dof_handler,
                                                         this_mpi_process);

    matrix_u.reinit (mpi_communicator,
                          dof_handler.n_dofs(),
                          dof_handler.n_dofs(),
                          n_local_dofs,
                          n_local_dofs,
                          dof_handler.max_couplings_between_dofs());

    matrix_v.reinit (mpi_communicator,
                          dof_handler.n_dofs(),
                          dof_handler.n_dofs(),
                          n_local_dofs,
                          n_local_dofs,
                          dof_handler.max_couplings_between_dofs());


    QGauss<dim>   quadrature_formula(3);
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;

    const unsigned int   n_q_points    = quadrature_formula.size();


    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    
    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

    int koko = 0;
    for (; cell!=endc; ++cell)
    {
      if (cell->subdomain_id() == this_mpi_process)
      {
        koko++;
        fe_values.reinit (cell);
        cell_matrix = 0;

        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j) 
              {
                cell_matrix(i,j) += (fe_values.shape_value (i, q_index) *
                                     fe_values.shape_value (j, q_index) *
                                     fe_values.JxW (q_index));

              }
            }

        cell->get_dof_indices (local_dof_indices);
        hanging_node_constraints
        .distribute_local_to_global(cell_matrix, 
                                    local_dof_indices,
                                    matrix_u); 
        hanging_node_constraints
        .distribute_local_to_global(cell_matrix, 
                                    local_dof_indices,
                                    matrix_v); 
      }
      
    }
    matrix_u.compress(VectorOperation::add);
    matrix_v.compress(VectorOperation::add);
  }

  template <int dim>
  void WaveEquation<dim>::run ()
  {
    setup_system();
    
    TimerOutput::Scope t(computing_timer, "run");
    TimerOutput::Scope twall(computing_timer_wall, "run");

    VectorTools::project (dof_handler, hanging_node_constraints, QGauss<dim>(3),
                          InitialValuesU<dim>(),
                          old_solution_u);
    VectorTools::project (dof_handler, hanging_node_constraints, QGauss<dim>(3),
                          InitialValuesV<dim>(),
                          old_solution_v);

    PETScWrappers::MPI::Vector tmp (mpi_communicator, solution_u.size(), solution_u.local_size());
    PETScWrappers::MPI::Vector forcing_terms (mpi_communicator, solution_u.size(), solution_u.local_size());

    PETScWrappers::MPI::Vector k_rhs (mpi_communicator, solution_u.size(), solution_u.local_size());

    for (unsigned int i=0; i<k_rhs.size(); ++i)
      k_rhs(i) = 0;
    k_rhs.compress (VectorOperation::add);

    for (timestep_number=1, time=time_step;
         time<=2;
         time+=time_step, ++timestep_number)
      {
        std::cout << "Time step " << timestep_number
                  << " at t=" << time
                  << std::endl;

        hehe ();

        mass_matrix.vmult (system_rhs, old_solution_u);

        mass_matrix.vmult (tmp, old_solution_v);
        system_rhs.add (time_step, tmp);

        laplace_matrix.vmult (tmp, old_solution_u);
        system_rhs.add (-theta * (1-theta) * time_step * time_step, tmp);

        forcing_terms = k_rhs;
        forcing_terms *= theta * time_step;

        forcing_terms.add ((1-theta) * time_step, k_rhs);

        system_rhs.add (theta * time_step, forcing_terms);

        {
          BoundaryValuesU<dim> boundary_values_u_function;
          boundary_values_u_function.set_time (time);

          std::map<types::global_dof_index,double> boundary_values;
          VectorTools::interpolate_boundary_values (dof_handler,
                                                    0,
                                                    boundary_values_u_function,
                                                    boundary_values);

          const types::global_dof_index n_local_dofs
            = DoFTools::count_dofs_with_subdomain_association (dof_handler,
                                                         this_mpi_process);
    
          matrix_u.copy_from(mass_matrix);
          matrix_u.add (laplace_matrix, theta * theta * time_step * time_step);
          MatrixTools::apply_boundary_values (boundary_values,
                                              matrix_u,
                                              solution_u,
                                              system_rhs, false);
        }
        solve_u ();


        laplace_matrix.vmult (system_rhs, solution_u);
        system_rhs *= -theta * time_step;

        mass_matrix.vmult (tmp, old_solution_v);
        system_rhs += tmp;

        laplace_matrix.vmult (tmp, old_solution_u);
        system_rhs.add (-time_step * (1-theta), tmp);

        system_rhs += forcing_terms;

        {
          BoundaryValuesV<dim> boundary_values_v_function;
          boundary_values_v_function.set_time (time);

          std::map<types::global_dof_index,double> boundary_values;
          VectorTools::interpolate_boundary_values (dof_handler,
                                                    0,
                                                    boundary_values_v_function,
                                                    boundary_values);

          const types::global_dof_index n_local_dofs
            = DoFTools::count_dofs_with_subdomain_association (dof_handler,
                                                         this_mpi_process);

          matrix_v.copy_from (mass_matrix);
          MatrixTools::apply_boundary_values (boundary_values,
                                              matrix_v,
                                              solution_v,
                                              system_rhs, false);
        }
        solve_v ();


        output_results ();


        pcout << "   Total energy: "
                  << (mass_matrix.matrix_norm_square (solution_v) +
                      laplace_matrix.matrix_norm_square (solution_u)) / 2
                  << std::endl;

        old_solution_u.compress (VectorOperation::add);
        old_solution_v.compress (VectorOperation::add);
        old_solution_u = solution_u;
        old_solution_v = solution_v;
      }
  }
}


// @sect3{The <code>main</code> function}

// What remains is the main function of the program. There is nothing here
// that hasn't been shown in several of the previous programs:
int main (int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace Step23;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      deallog.depth_console (0);

      WaveEquation<2> wave_equation_solver;
      wave_equation_solver.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
