/** ------------------------------------------------------------------------------ *
 * author(s)   : Merlind Schotte (schotte@zib.de)                                  *
 * institution : Zuse Institute Berlin (ZIB)                                       *
 * project     : HighPerMeshes (BMBF)                                              *
 *                                                                                 *
 * Description:                                                                    *
 * Setting options of 2D monodomain example (input values).                        *
 *                                                                                 *
 * last change: 17.12.20                                                           *
 * -----------------------------------------------------------------------------  **/

#include </usr/include/c++/7/iostream>
#include <HighPerMeshes.hpp>

//!
//! \brief Set start settings.
//! \todo add output file with parameter overview
//!
template<typename FloatT>
void SetStartValues(int option, FloatT& h, FloatT& a, FloatT& b, FloatT& eps, FloatT& sigma, FloatT& u0L,
                    FloatT& u0R, FloatT& w0L, FloatT& w0R/*, ofstream& file*/)
{
    if (option == 0)
    {
        // input options for small and bigger meshs
        h     = 0.5;//0.2; // step size
        a     = 0.1;
        b     = 1e-4;
        eps   = 5e-4;
        sigma = -0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else if (option == 1)
    {
        // input options for small mesh 5x5 (config.cfg -> mesh2D_test5x5.am)
        h     = 0.015; // step size
        a     = -0.1;
        b     = 0.008;
        eps   = 0.1;
        sigma = -0.1;
        u0L   = 0.1; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.1;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else if (option == 2)
    {
        // input options for bigger mesh 100x100 (config.cfg -> mesh2D.am)
        h     = 0.4; // step size
        a     = 0.1;
        b     = 1e-4;
        eps   = 0.005;
        sigma = -0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else if (option == 3)
    {
        // input options for bigger mesh 100x100 (config.cfg -> mesh2D.am)
        h     = 0.4; // step size
        a     = 0.1;
        b     = 1e-4;
        eps   = 1e-4;
        sigma = -0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else if (option == 4)
    {
        h     = 0.00001; // time step size h <= 0.0005
        a     = 0.7;
        b     = 1e-4;
        eps   = 4.0;//0.5; //1;
        sigma = 10; // diffusion tensor sigma <= 0.1
        u0L   = 1.F; //1.F; // values for start vector u
        u0R   = 0.F; // values for start vector u
        w0L   = 0.F;  // values for start vector w
        w0R   = 0.F;  // values for start vector w
    }
    else if (option == 5)
    {
        // input options for bigger mesh 100x100 (config.cfg -> mesh2D.am)
        h     = 0.5; // step size
        a     = 0.1;
        b     = 1e-4;
        eps   = 1e-3;//0.001;
        sigma = -0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else if (option == 6)
    {
        // input options for bigger mesh 100x100 (config.cfg -> mesh2D.am)
        h     = 0.005; // step size
        a     = 0.7;
        b     = 1e-3;
        eps   = 1e-3;//0.001;
        sigma = -0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else
        printf("There is no option choosen for start value settings. Create your own option or choose one of the existing.");

    // add parameter to .txt file -> TODO: fix bug
    //    string fileName = "testParameterfile.txt";
    //    WriteParameterInfoIntoFile(file, fileName, h, "h");
    //    WriteParameterInfoIntoFile(file, fileName, a, "a");
    //    WriteParameterInfoIntoFile(file, fileName, b, "b");
    //    WriteParameterInfoIntoFile(file, fileName, eps, "eps");
    //    WriteParameterInfoIntoFile(file, fileName, sigma, "sigma");

    return;
}
