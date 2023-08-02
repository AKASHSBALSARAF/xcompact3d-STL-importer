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
                inum = 0
                if array[0, j, k]:
                    xi[inum, j, k] = -dd[0]

                for i in range(shape[0] - 1):
                    if not array[i, j, k] and array[i + 1, j, k]:
                        xi[inum, j, k] = dd[1] * i + dd[1] / 2
                    elif array[i, j, k] and not array[i + 1, j, k]:
                        xf[inum, j, k] = dd[1] * i + dd[1] / 2
                        inum += 1

                if array[-1, j, k]:
                    xf[inum, j, k] = lx + dd[0]

        return xi, xf

    elif dim == 'y':
        yi = np.zeros((nobjmax, shape[0], shape[2]))
        yf = np.zeros((nobjmax, shape[0], shape[2]))

        for k in range(shape[2]):
            for i in range(shape[0]):
                jnum = 0
                if array[i, 0, k]:
                    yi[jnum, i, k] = -dd[2]

                for j in range(shape[1] - 1):
                    if not array[i, j, k] and array[i, j + 1, k]:
                        yi[jnum, i, k] = dd[3] * j + dd[3] / 2
                    elif array[i, j, k] and not array[i, j + 1, k]:
                        yf[jnum, i, k] = dd[3] * j + dd[3] / 2
                        jnum += 1

                if array[i, -1, k]:
                    yf[jnum, i, k] = ly + dd[2]

        return yi, yf

    else:
        zi = np.zeros((nobjmax, shape[0], shape[1]))
        zf = np.zeros((nobjmax, shape[0], shape[1]))

        for j in range(shape[1]):
            for i in range(shape[0]):
                knum = 0
                if array[i, j, 0]:
                    zi[knum, i, j] = -dd[4]

                for k in range(shape[2] - 1):
                    if not array[i, j, k] and array[i, j, k + 1]:
                        zi[knum, i, j] = dd[5] * k + dd[5] / 2
                    elif array[i, j, k] and not array[i, j, k + 1]:
                        zf[knum, i, j] = dd[5] * k + dd[5] / 2
                        knum += 1

                if array[i, j, -1]:
                    zf[knum, i, j] = ly + dd[4]

        return zi, zf

def fixBugs(ep,rafep,dim,shape,nobj,nobjraf, nraf, nobjmax, dimi, dimf):
    ibugs = np.where(np.not_equal(nobj,nobjraf))
    print(ibugs)
    if(dim=='x'):
        for l in len(ibugs[0]):
            iobj = -1

            if (ep[0,ibugs[0][l],ibugs[1][l]]):
                iobj += 1 

            for i in range(shape[0]-1):

                if(not ep[i,ibugs[0][l],ibugs[1][l]] and ep[i+1,ibugs[0][l],ibugs[1][l]]):
                    iobj += 1

                elif(not ep[i,ibugs[0][l],ibugs[1][l]] and not ep[i+1,ibugs[0][l],ibugs[1][l]]):
                    iflu = 1

                elif(ep[i,ibugs[0][l],ibugs[1][l]] and ep[i+1,ibugs[0][l],ibugs[1][l]]):
                    isol = 1

                for iraf in range(nraf-1):

                    if not rafep[iraf+nraf*(i), ibugs[0][l], ibugs[1][l]] and rafep[iraf+nraf*(i)+1, ibugs[0][l], ibugs[1][l]]:
                        idebraf = iraf + nraf*(i)

                    if rafep[iraf+nraf*(i), ibugs[0][l], ibugs[1][l]] and not rafep[iraf+nraf*(i)+1, ibugs[0][l], ibugs[1][l]]:
                        ifinraf = iraf + nraf*(i)

                if idebraf != 0 and ifinraf != 0 and idebraf < ifinraf and iflu == 1:
                    iobj += 1

                    for ii in range(iobj, nobjmax-1):

                        dimi[ii,ibugs[0][l], ibugs[1][l]] = dimi[ii+1,ibugs[0][l], ibugs[1][l]]
                        dimf[ii,ibugs[0][l], ibugs[1][l]] = dimf[ii+1,ibugs[0][l], ibugs[1][l]] 

                    iobj -= 1

                if idebraf != 0 and ifinraf != 0 and idebraf > ifinraf and isol == 1:
                    iobj += 1
                    for ii in range(iobj, nobjmax-1):
                        dimi[ii,ibugs[0][l],ibugs[1][l]] = dimi[ii+1,ibugs[0][l],ibugs[1][l]]
                    
                    iobj -= 1
                    for ii in range(iobj, nobjmax-1):
                        dimf[ii,ibugs[0][l],ibugs[1][l]] = dimf[ii+1,ibugs[0][l],ibugs[1][l]]

                idebraf = 0
                ifinraf = 0
                iflu = 0
                isol = 0

    elif(dim=='y'):
        for l in len(ibugs[0]):
            iobj = -1

            if (ep[ibugs[0][l],0,ibugs[1][l]]):
                iobj += 1 

            for j in range(shape[1]-1):

                if(not ep[ibugs[0][l],j,ibugs[1][l]] and ep[ibugs[0][l],j+1,ibugs[1][l]]):
                    iobj += 1

                elif(not ep[ibugs[0][l],j,ibugs[1][l]] and not ep[ibugs[0][l],j+1,ibugs[1][l]]):
                    iflu = 1

                elif(ep[ibugs[0][l],j,ibugs[1][l]] and ep[ibugs[0][l],j+1,ibugs[1][l]]):
                    isol = 1

                for iraf in range(nraf-1):

                    if not rafep[ibugs[0][l], iraf+nraf*(j), ibugs[1][l]] and rafep[ibugs[0][l], iraf+nraf*(j)+1, ibugs[1][l]]:
                        idebraf = iraf + nraf*(j)

                    if rafep[ibugs[0][l], iraf+nraf*(j-1), ibugs[1][l]] and not rafep[ibugs[0][l], iraf+nraf*(j)+1, ibugs[1][l]]:
                        ifinraf = iraf + nraf*(j)

                if idebraf != 0 and ifinraf != 0 and idebraf < ifinraf and iflu == 1:
                    iobj += 1

                    for ii in range(iobj, nobjmax-1):

                        dimi[ii,ibugs[0][l], ibugs[1][l]] = dimi[ii+1,ibugs[0][l], ibugs[1][l]]
                        dimf[ii,ibugs[0][l], ibugs[1][l]] = dimf[ii+1,ibugs[0][l], ibugs[1][l]] 

                    iobj -= 1

                if idebraf != 0 and ifinraf != 0 and idebraf > ifinraf and isol == 1:
                    iobj += 1
                    for ii in range(iobj, nobjmax-1):
                        dimi[ii,ibugs[0][l],ibugs[1][l]] = dimi[ii+1,ibugs[0][l],ibugs[1][l]]
                    
                    iobj -= 1
                    for ii in range(iobj, nobjmax-1):
                        dimf[ii,ibugs[0][l],ibugs[1][l]] = dimf[ii+1,ibugs[0][l],ibugs[1][l]]

                idebraf = 0
                ifinraf = 0
                iflu = 0
                isol = 0

    else:
        for l in len(ibugs[0]):
            iobj = -1

            if (ep[ibugs[0][l],ibugs[1][l],0]):
                iobj += 1 

            for k in range(shape[2]-1):

                if(not ep[ibugs[0][l],ibugs[1][l],k] and ep[ibugs[0][l],ibugs[1][l],k+1]):
                    iobj += 1

                elif(not ep[ibugs[0][l],ibugs[1][l],k] and not ep[ibugs[0][l],ibugs[1][l],k+1]):
                    iflu = 1

                elif(ep[ibugs[0][l],ibugs[1][l],k] and ep[ibugs[0][l],ibugs[1][l],k+1]):
                    isol = 1

                for iraf in range(nraf-1):

                    if not rafep[ibugs[0][l], ibugs[1][l], iraf+nraf*(k)] and rafep[ibugs[0][l],ibugs[1][l], iraf+nraf*(k)+1]:
                        idebraf = iraf + nraf*(k)

                    if rafep[ibugs[0][l], ibugs[1][l], iraf+nraf*(k)] and not rafep[ibugs[0][l],ibugs[1][l], iraf+nraf*(k)+1]:
                        ifinraf = iraf + nraf*(k)

                if idebraf != 0 and ifinraf != 0 and idebraf < ifinraf and iflu == 1:
                    iobj += 1

                    for ii in range(iobj, nobjmax-1):

                        dimi[ii,ibugs[0][l], ibugs[1][l]] = dimi[ii+1,ibugs[0][l], ibugs[1][l]]
                        dimf[ii,ibugs[0][l], ibugs[1][l]] = dimf[ii+1,ibugs[0][l], ibugs[1][l]] 
                    iobj -= 1

                if idebraf != 0 and ifinraf != 0 and idebraf > ifinraf and isol == 1:
                    iobj += 1
                    for ii in range(iobj, nobjmax-1):
                        dimi[ii,ibugs[0][l],ibugs[1][l]] = dimi[ii+1,ibugs[0][l],ibugs[1][l]]
                    
                    iobj -= 1
                    for ii in range(iobj, nobjmax-1):
                        dimf[ii,ibugs[0][l],ibugs[1][l]] = dimf[ii+1,ibugs[0][l],ibugs[1][l]]

                idebraf = 0
                ifinraf = 0
                iflu = 0
                isol = 0

    return dimi, dimf
                

def verify(ep, dim, shape, nobjmax, npif, izap):
    ising = 0

    if dim == 'x':
        nxipif = np.full((nobjmax + 1, shape[1], shape[2]), npif, dtype=np.int64)
        nxfpif = np.full((nobjmax + 1, shape[1], shape[2]), npif, dtype=np.int64)

        mask = np.concatenate((np.zeros((1, shape[1], shape[2]), dtype=bool), ~ep[:-1, :, :]), axis=0)
        mask_diff = np.diff(mask, axis=0)

        i_obj_start, = np.where(mask_diff)
        i_obj_end, = np.where(mask_diff == -1)

        inum = -1
        iflu = -1

        for start, end in zip(i_obj_start, i_obj_end):
            if not mask[start, 0, 0]:
                iflu += 1

            if not mask[end, 0, 0]:
                if iflu - izap < npif:
                    nxipif[inum, :, :] = iflu - izap
                    ising += 1
                iflu = 0

            iflu_vec = np.cumsum(~mask[start:end + 1, :, :], axis=0)

            is_start = mask[start, :, :]
            is_end = mask[end, :, :]

            inum_vec = np.where(~is_start, iflu_vec, 0)

            inum += inum_vec[-1, :, :]

            if iflu_vec.shape[0] > 1:
                nxfpif[inum_vec[:-1, :, :]] = iflu_vec[:-1, :, :] - izap

        return nxipif, nxfpif

    elif dim == 'y':
        nyipif = np.full((nobjmax + 1, shape[0], shape[2]), npif, dtype=np.int64)
        nyfpif = np.full((nobjmax + 1, shape[0], shape[2]), npif, dtype=np.int64)

        mask = np.concatenate((np.zeros((shape[0], 1, shape[2]), dtype=bool), ~ep[:, :-1, :]), axis=1)
        mask_diff = np.diff(mask, axis=1)

        i_obj_start, = np.where(mask_diff)
        i_obj_end, = np.where(mask_diff == -1)

        inum = -1
        iflu = -1

        for start, end in zip(i_obj_start, i_obj_end):
            if not mask[0, start, 0]:
                iflu += 1

            if not mask[0, end, 0]:
                if iflu - izap < npif:
                    nyipif[inum, :, :] = iflu - izap
                    ising += 1
                iflu = 0

            iflu_vec = np.cumsum(~mask[:, start:end + 1, :], axis=1)

            is_start = mask[:, start, :]
            is_end = mask[:, end, :]

            inum_vec = np.where(~is_start, iflu_vec, 0)

            inum += inum_vec[:, -1, :]

            if iflu_vec.shape[1] > 1:
                nyfpif[inum_vec[:, :-1, :]] = iflu_vec[:, :-1, :] - izap

        return nyipif, nyfpif

    elif dim == 'z':
        nzipif = np.full((nobjmax + 1, shape[0], shape[1]), npif, dtype=np.int64)
        nzfpif = np.full((nobjmax + 1, shape[0], shape[1]), npif, dtype=np.int64)

        mask = np.concatenate((np.zeros((shape[0], shape[1], 1), dtype=bool), ~ep[:, :, :-1]), axis=2)
        mask_diff = np.diff(mask, axis=2)

        i_obj_start, = np.where(mask_diff)
        i_obj_end, = np.where(mask_diff == -1)

        inum = -1
        iflu = -1

        for start, end in zip(i_obj_start, i_obj_end):
            if not mask[0, 0, start]:
                iflu += 1

            if not mask[0, 0, end]:
                if iflu - izap < npif:
                    nzipif[inum, :, :] = iflu - izap
                    ising += 1
                iflu = 0

            iflu_vec = np.cumsum(~mask[:, :, start:end + 1], axis=2)

            is_start = mask[:, :, start]
            is_end = mask[:, :, end]

            inum_vec = np.where(~is_start, iflu_vec, 0)

            inum += inum_vec[:, :, -1]

            if iflu_vec.shape[2] > 1:
                nzfpif[inum_vec[:, :, :-1]] = iflu_vec[:, :, :-1] - izap

        return nzipif, nzfpif
