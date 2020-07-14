/** ------------------------------------------------------------------------------ *
 * author(s)   : Merlind Schotte (schotte@zib.de)                                  *
 * institution : Zuse Institute Berlin (ZIB)                                       *
 * project     : HighPerMeshes (BMBF)                                              *
 *                                                                                 *
 * Description:                                                                    *
 * Implementation of some simple functions.                                        *
 *                                                                                 *
 * last change: 21.11.19                                                           *
 * -----------------------------------------------------------------------------  **/

#include <iostream>
#include <HighPerMeshes.hpp>

//!
//! \return gradients (p1-elements, dim = 3)
//!
auto GetGradientsDSL()
{
    HPM::dataType::Matrix<float,4,3> gradientsDSL;
    gradientsDSL[0][0]= -1; gradientsDSL[0][1]= -1; gradientsDSL[0][2]= -1;
    gradientsDSL[1][0]=  1; gradientsDSL[1][1]=  0; gradientsDSL[1][2]=  0;
    gradientsDSL[2][0]=  0; gradientsDSL[2][1]=  1; gradientsDSL[2][2]=  0;
    gradientsDSL[3][0]=  0; gradientsDSL[3][1]=  0; gradientsDSL[3][2]=  1;
    return gradientsDSL;
}

//!
//! \return gradients (p1-elements, dim = 2)
//!
auto GetGradients2DP1()
{
    HPM::dataType::Matrix<float,3,2> gradientsDSL;
    gradientsDSL[0][0]= -1; gradientsDSL[0][1]= -1;
    gradientsDSL[1][0]=  1; gradientsDSL[1][1]=  0;
    gradientsDSL[2][0]=  0; gradientsDSL[2][1]=  1;
    return gradientsDSL;
}
