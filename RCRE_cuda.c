//
// Created by Mads Bjoern on 01/09/2019.
//

#include "RCRE_cuda.h"
#include "RCRE_constants.h"

#ifdef RCRE_CUDA

RCRE_color *RCRE_color_toCUDA(RCRE_color *host) {
    RCRE_color *device;
    cudaMalloc((void**)&device, sizeof(RCRE_color));
    cudaMemcpy(device, host, sizeof(RCRE_color), cudaMemcpyHostToDevice);
    return device;
}

RCRE_convexPolyhedron *RCRE_convexPolyhedron_toCUDA(RCRE_convexPolyhedron *host) {
    RCRE_convexPolyhedron *device;
    cudaMalloc((void**)&device, sizeof(RCRE_convexPolyhedron));

    RCRE_convexPolyhedron temp = {0};
    temp.nTriangles = host->nTriangles;
    temp.boundingSphereRadius = host->boundingSphereRadius;
    temp.centerPoint = RCRE_point3D_toCUDA(host->centerPoint);
    cudaMalloc((void**)&temp.triangles, sizeof(RCRE_triangle3D *) * host->nTriangles);
    RCRE_triangle3D **tempList = (RCRE_triangle3D **)malloc(sizeof(RCRE_triangle3D *) * host->nTriangles);
    for (int i = 0; i < host->nTriangles; i ++) {
        tempList[i] = RCRE_triangle3D_toCUDA(host->triangles[i]);
    }
    cudaMemcpy(temp.triangles, &tempList, sizeof(RCRE_triangle3D *) * host->nTriangles, cudaMemcpyHostToDevice);
    free(tempList);

    cudaMemcpy(device, &temp, sizeof(RCRE_convexPolyhedron), cudaMemcpyHostToDevice);
    return device;
}

RCRE_model3D *RCRE_model3D_toCUDA(RCRE_model3D *host) {
    RCRE_model3D *device;
    cudaMalloc((void**)&device, sizeof(RCRE_model3D));

    RCRE_model3D tempModel = {0};
    tempModel.centerPoint = RCRE_point3D_toCUDA(host->centerPoint);
    tempModel.color = RCRE_color_toCUDA(host->color);
    tempModel.nEntries = host->nEntries;
    tempModel.nEntriesSpace = host->nEntriesSpace;
    tempModel.reflectivity = host->reflectivity;
    tempModel.refractiveIndex = host->refractiveIndex;
    tempModel.transparency = host->transparency;
    tempModel.transformationMatrix = RCRE_transformationMatrix_toCUDA(host->transformationMatrix);
    tempModel.inverseTransformationMatrix = RCRE_transformationMatrix_toCUDA(host->inverseTransformationMatrix);
    cudaMalloc((void**)&tempModel.entries, sizeof(RCRE_model3D_entry *) * host->nEntries);
    RCRE_model3D_entry **tempEntries = (RCRE_model3D_entry **)malloc(sizeof(RCRE_model3D_entry *) * host->nEntries);
    for (int i = 0; i < host->nEntries; i ++) {
        RCRE_model3D_entry *deviceEntry;
        cudaMalloc((void**)&deviceEntry, sizeof(RCRE_model3D_entry));

        RCRE_model3D_entry tempEntry = {0};
        tempEntry.negative = host->entries[i]->negative;
        tempEntry.datatype = host->entries[i]->datatype;
        tempEntry.transformationMatrix = RCRE_transformationMatrix_toCUDA(host->entries[i]->transformationMatrix);
        tempEntry.inverseTransformationMatrix = RCRE_transformationMatrix_toCUDA(host->entries[i]->inverseTransformationMatrix);
        if (host->entries[i]->datatype == RCRE_model3D_DATATYPE_SPHERE) {
            tempEntry.data = RCRE_sphere_toCUDA(host->entries[i]->data);
        } else if (host->entries[i]->datatype == RCRE_model3D_DATATYPE_CONVEX_POLYHEDRON) {
            tempEntry.data = RCRE_convexPolyhedron_toCUDA(host->entries[i]->data);
        }

        cudaMemcpy(deviceEntry, &tempEntry, sizeof(RCRE_model3D_entry), cudaMemcpyHostToDevice);
        tempEntries[i] = deviceEntry;
    }
    cudaMemcpy(tempModel.entries, &tempEntries, sizeof(RCRE_model3D_entry *) * host->nEntries, cudaMemcpyHostToDevice);
    free(tempEntries);

    cudaMemcpy(device, &tempModel, sizeof(RCRE_model3D), cudaMemcpyHostToDevice);
    return device;
}

RCRE_point3D *RCRE_point3D_toCUDA(RCRE_point3D *host) {
    RCRE_point3D *device;
    cudaMalloc((void**)&device, sizeof(RCRE_point3D));
    cudaMemcpy(device, host, sizeof(RCRE_point3D), cudaMemcpyHostToDevice);
    return device;
}

RCRE_sphere *RCRE_sphere_toCUDA(RCRE_sphere *host) {
    RCRE_sphere *device;
    cudaMalloc((void**)&device, sizeof(RCRE_sphere));
    RCRE_sphere temp = {0};
    temp.center = RCRE_point3D_toCUDA(host->center);
    temp.radius = host->radius;
    cudaMemcpy(device, &temp, sizeof(RCRE_sphere), cudaMemcpyHostToDevice);
    return device;
}

RCRE_transformationMatrix *RCRE_transformationMatrix_toCUDA(RCRE_transformationMatrix *host) {
    RCRE_transformationMatrix *device;
    cudaMalloc((void**)&device, sizeof(RCRE_transformationMatrix));
    cudaMemcpy(device, host, sizeof(RCRE_transformationMatrix), cudaMemcpyHostToDevice);
    return device;
}

RCRE_triangle3D *RCRE_triangle3D_toCUDA(RCRE_triangle3D *host) {
    RCRE_triangle3D *device;
    cudaMalloc((void**)&device, sizeof(RCRE_triangle3D));
    RCRE_triangle3D temp = {0};
    temp.p1 = RCRE_point3D_toCUDA(host->p1);
    temp.p2 = RCRE_point3D_toCUDA(host->p2);
    temp.p3 = RCRE_point3D_toCUDA(host->p3);
    cudaMemcpy(device, &temp, sizeof(RCRE_triangle3D), cudaMemcpyHostToDevice);
    return device;
}

#endif