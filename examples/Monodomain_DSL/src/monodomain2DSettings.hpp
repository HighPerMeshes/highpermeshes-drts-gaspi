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
    if (option == 0) //5x5 Mesh
    {
        h     = 0.2; // step size for time
        a     = 0.1;
        b     = 1e-4;
        eps   = 5e-4;
        sigma = -0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else if (option == 1) //20x20 Mesh with 100 iterations
    {
        h     = 0.6; // step size for time
        a     = 0.1;
        b     = 1e-4;
        eps   = 0.005;
        sigma = -0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else if (option == 2)
    {
        // input options for bigger mesh 100x100 (config.cfg -> mesh2D.am)
        h     = 0.03; // step size
        a     = 0.1;
        b     = 1e-4;
        eps   = 0.05;//0.005;
        sigma = -0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 1.F;  // values for start vector w on \Omega_1 and \Omega_2
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
