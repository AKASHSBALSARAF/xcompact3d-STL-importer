import numpy as np

def obj_count(array, dim, shape):
    if dim == 'x':
        nobjx = np.int64(array[0, :, :])
        for i in range(shape[0] - 1):
            mask1 = ~array[i, :, :] & array[i + 1, :, :]
            nobjx[mask1] += 1
        return nobjx
    elif dim == 'y':
        nobjy = np.int64(array[:, 0, :])
        for j in range(shape[1] - 1):
            mask2 = ~array[:, j, :] & array[:, j + 1, :]
            nobjy[mask2] += 1
        return nobjy
    else:
        nobjz = np.int64(array[:, :, 0])
        for k in range(shape[2] - 1):
            mask3 = ~array[:, :, k] & array[:, :, k + 1]
            nobjz[mask3] += 1
        return nobjz

def get_boundaries(array, dim, shape, lx, ly, lz, nobjmax, dd):
    if dim == 'x':
        xi = np.zeros((nobjmax, shape[1], shape[2]))
        xf = np.zeros((nobjmax, shape[1], shape[2]))

        for k in range(shape[2]):
            for j in range(shape[1]):
                mask = array[:, j, k]
                diff = np.diff(mask)

                start_indices = np.where(diff == 1)[0] + 1
                end_indices = np.where(diff == -1)[0]

                if mask[0]:
                    xi[0, j, k] = -dd[0]
                xi[1:len(start_indices) + 1, j, k] = (start_indices - 0.5) * dd[1]
                xf[:len(end_indices) + 1, j, k] = (end_indices + 0.5) * dd[1]
                xf[len(end_indices) + 1:, j, k] = lx + dd[0]

        return xi, xf

    elif dim == 'y':
        yi = np.zeros((nobjmax, shape[0], shape[2]))
        yf = np.zeros((nobjmax, shape[0], shape[2]))

        for k in range(shape[2]):
            for i in range(shape[0]):
                mask = array[i, :, k]
                diff = np.diff(mask)

                start_indices = np.where(diff == 1)[0] + 1
                end_indices = np.where(diff == -1)[0]

                if mask[0]:
                    yi[0, i, k] = -dd[2]
                yi[1:len(start_indices) + 1, i, k] = (start_indices - 0.5) * dd[3]
                yf[:len(end_indices) + 1, i, k] = (end_indices + 0.5) * dd[3]
                yf[len(end_indices) + 1:, i, k] = ly + dd[2]

        return yi, yf

    elif dim == 'z':
        zi = np.zeros((nobjmax, shape[0], shape[1]))
        zf = np.zeros((nobjmax, shape[0], shape[1]))

        for j in range(shape[1]):
            for i in range(shape[0]):
                mask = array[i, j, :]
                diff = np.diff(mask)

                start_indices = np.where(diff == 1)[0] + 1
                end_indices = np.where(diff == -1)[0]

                if mask[0]:
                    zi[0, i, j] = -dd[4]
                zi[1:len(start_indices) + 1, i, j] = (start_indices - 0.5) * dd[5]
                zf[:len(end_indices) + 1, i, j] = (end_indices + 0.5) * dd[5]
                zf[len(end_indices) + 1:, i, j] = ly + dd[4]

        return zi, zf

    else:
        raise ValueError("Invalid dimension. Supported dimensions are 'x', 'y', and 'z'.")

import numpy as np

def fixBugs(ep, rafep, dim, shape, nobj, nobjraf, nraf, nobjmax, dimi, dimf):
    ibugs = np.where(np.not_equal(nobj, nobjraf))
    
    for l in range(len(ibugs[0])):
        i, j, k = ibugs[0][l], ibugs[1][l], ibugs[2][l]

        if dim == 'x':
            dim_arr = dimi[:, i, j]
            dim_arr_f = dimf[:, i, j]
            ep_dim = ep[:, i, j]
            rafep_dim = rafep[:, i, j]
        elif dim == 'y':
            dim_arr = dimi[:, i, j]
            dim_arr_f = dimf[:, i, j]
            ep_dim = ep[i, :, j]
            rafep_dim = rafep[i, :, j]
        elif dim == 'z':
            dim_arr = dimi[:, i, j]
            dim_arr_f = dimf[:, i, j]
            ep_dim = ep[i, j, :]
            rafep_dim = rafep[i, j, :]
        else:
            raise ValueError("Invalid dimension. Supported dimensions are 'x', 'y', and 'z'.")

        iobj = -1
        if ep_dim[0]:
            iobj += 1

        for idx in range(shape[0] - 1):
            if not ep_dim[idx] and ep_dim[idx + 1]:
                iobj += 1

        if dim == 'x':
            rafep_dim = rafep_dim[:nraf * (shape[0] - 1)]
        elif dim == 'y':
            rafep_dim = rafep_dim[:nraf * (shape[1] - 1)]
        elif dim == 'z':
            rafep_dim = rafep_dim[:nraf * (shape[2] - 1)]

        idebraf = np.where(np.logical_and(~rafep_dim, np.roll(rafep_dim, -1)))[0]
        ifinraf = np.where(np.logical_and(rafep_dim, ~np.roll(rafep_dim, -1)))[0]

        if idebraf.size != 0 and ifinraf.size != 0 and idebraf[0] < ifinraf[0] and np.any(ep_dim == 0):
            iobj += 1
            dim_arr[iobj:] = dim_arr[iobj + 1:]
            dim_arr_f[iobj:] = dim_arr_f[iobj + 1:]

        if idebraf.size != 0 and ifinraf.size != 0 and idebraf[0] > ifinraf[0] and np.any(ep_dim == 1):
            iobj += 1
            dim_arr[iobj:] = dim_arr[iobj + 1:]

        dim_arr[nobjmax - 1] = dim_arr[iobj]
        dim_arr_f[nobjmax - 1] = dim_arr_f[iobj]

    return dimi, dimf               
