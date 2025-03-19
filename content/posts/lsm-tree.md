---
title: "Detailed Explanation of Bimocq Code"
date: "2025-03-17"
summary: "An interpretation of Bimocq opensource code"
description: "An LSM Tree overview and Java implementation."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
fediverse: "@username@instance.url"
---

## introduction

<img src="/images/teaser.jpg" alt="img1" width="100%">

Bimocq is proposed by the paper *Efficient and Conservative Fluids with Bidirectional Mapping* published on *ACM Transaction on Graphics* 2019, which is a Euler fluid simulator with GPU acceleration. This method  achieves high fidelity Turbulence simulation while remaining remarkable performance. As a result, it becomes an indispensable lesson for fluid simulation learners.

However, due to the complexity of this method,  starters may find it difficult to completely understand it. By interpreting the code ,we can look into the technical detail of this method.

## mapping update 

```
    VelocityAdvector.updateMapping(_un, _vn, _wn, cfldt, dt);
    ScalarAdvector.updateMapping(_un, _vn, _wn, cfldt, dt);
    cout << "[ Update Mapping Done! ]" << endl;
```

First we come to the "updateMapping" function.  Bimocq includes a forward and a backward mapping, both come from the Lagrange perspective.

The forward mapping, as its literal meaning suggests, is the mapping of the particle from the initial position to the current position.

<img src="/images/img1.png" alt="img1" width="30%">

And the inverse mapping is exactly the opposite.

<img src="/images/img2.png" alt="img2" width="30%">

So the first part of the solver is updating the mapping functions using the velocity calculated in the last iteration:_un, _vn, _wn

We skip several levels of function nesting, and look directly into the Cuda solving function.

``` 
__global__ void DMC_backward_kernel(float *u, float *v, float *w,
                                    float *x_in, float *y_in, float *z_in,
                                    float *x_out, float *y_out, float *z_out,
                                    float h, int ni, int nj, int nk, float substep)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i > 1 && i<ni-2 && j > 1 && j<nj-2 && k > 1 && k<nk-2)
    {
        float3 point = make_float3(h*float(i),h*float(j),h*float(k));

        float3 vel = getVelocity(u, v, w, h, ni, nj, nk, point);

        float temp_x = (vel.x > 0)? point.x - h: point.x + h;
        float temp_y = (vel.y > 0)? point.y - h: point.y + h;
        float temp_z = (vel.z > 0)? point.z - h: point.z + h;
        float3 temp_point = make_float3(temp_x, temp_y, temp_z);
        float3 temp_vel = getVelocity(u, v, w, h, ni, nj, nk, temp_point);

        float a_x = (vel.x - temp_vel.x) / (point.x - temp_point.x);
        float a_y = (vel.y - temp_vel.y) / (point.y - temp_point.y);
        float a_z = (vel.z - temp_vel.z) / (point.z - temp_point.z);

        float new_x = (fabs(a_x) > 1e-4)? point.x - (1 - exp(-a_x*substep))*vel.x/a_x : point.x - vel.x*substep;
        float new_y = (fabs(a_y) > 1e-4)? point.y - (1 - exp(-a_y*substep))*vel.y/a_y : point.y - vel.y*substep;
        float new_z = (fabs(a_z) > 1e-4)? point.z - (1 - exp(-a_z*substep))*vel.z/a_z : point.z - vel.z*substep;
        float3 pointnew = make_float3(new_x, new_y, new_z);

        x_out[index] = sample_buffer(x_in,ni,nj,nk,h,make_float3(0.0,0.0,0.0),pointnew);
        y_out[index] = sample_buffer(y_in,ni,nj,nk,h,make_float3(0.0,0.0,0.0),pointnew);
        z_out[index] = sample_buffer(z_in,ni,nj,nk,h,make_float3(0.0,0.0,0.0),pointnew);
    }
    __syncthreads();
}
```

Here,x_in,y_in,z_in are the previews backward mapping. And ''getVelocity'' is a function that interpolates velocity from the grid.

All the calculation below is the ''Dual mesh characteristic'' method, which is a improved semi-Lagrange advection method with second-order accuracy. You can find detailed explanation in the paper.

Once we obtain the advected particle position x_in, we obtain new backward mapping by interpolating  in the backward mapping grid using x_in.



```
__global__ void forward_kernel(float *u, float *v, float *w,
                            float *x_fwd, float *y_fwd, float *z_fwd,
                            float h, int ni, int nj, int nk, float cfldt, float dt)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i > 1 && i<ni-2 && j > 1 && j<nj-2 && k > 1 && k<nk-2)
    {
        float3 point = make_float3(x_fwd[index], y_fwd[index], z_fwd[index]);
        float3 pointout = trace(u,v,w,h,ni,nj,nk,cfldt,dt,point);
        x_fwd[index] = pointout.x;
        y_fwd[index] = pointout.y;
        z_fwd[index] = pointout.z;
    }
    __syncthreads();
}
```

The forward solving is much simpler. We simply do the forward tracing by calling the ''trace'' function. This function implements Runge-Kutta methods to achieve third-order accuracy in forward tracing. You can look into the code yourself to find the detailed implementation.



## Advection

```
void BimocqSolver::semilagAdvect(float cfldt, float dt)
{
    // NOTE: TO SAVE TRANSFER TIME, NEED U,V,W BE STORED IN GPU.U, GPU.V, GPU.W ALREADY
    // semi-lagrangian advected velocity will be stored in gpu.du, gpu.dv, gpu.dw
    // negate dt for tracing back
    cudaMemcpy(gpuSolver->u_src, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(cfldt, -dt);
    gpuSolver->copyDeviceToHost(_utemp, gpuSolver->u_host, gpuSolver->du);
    gpuSolver->copyDeviceToHost(_vtemp, gpuSolver->v_host, gpuSolver->dv);
    gpuSolver->copyDeviceToHost(_wtemp, gpuSolver->w_host, gpuSolver->dw);
    // semi-lagrangian advect any other fluid fields
    // reuse gpu.du, gpu.dv to save GPU buffer
    // copy field to gpu.dv for semi-lagrangian advection
    // advect density
    gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->semilagAdvectField(cfldt, -dt);
    gpuSolver->copyDeviceToHost(_rhotemp, gpuSolver->x_host, gpuSolver->du);
    // advect Temperature
    gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->semilagAdvectField(cfldt, -dt);
    gpuSolver->copyDeviceToHost(_Ttemp, gpuSolver->x_host, gpuSolver->du);
}
```

Next, the solver call the ''semilagAdvect" function to do one-step semi-lagrangian advection. The advection result of velocity,density and temperature is stored in _utemp,_vtemp,_wtemp,_rhotemp and _Ttemp.

It is worth mentioning that semilag advection results will not be used to update the fluid state. In stead, it will only be used in "blendBoundary" function, which you can investigate yourself.

```
void MapperBase::advectVelocity(buffer3Df &un, buffer3Df &vn, buffer3Df &wn,
                                const buffer3Df &u_init, const buffer3Df &v_init, const buffer3Df &w_init,
                                const buffer3Df &u_prev, const buffer3Df &v_prev, const buffer3Df &w_prev)
{
    // forward and backward mapping are already in gpu.x_out and gpu.x_in correspondingly
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_x, gpuSolver->x_host, gpuSolver->x_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_y, gpuSolver->y_host, gpuSolver->y_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_z, gpuSolver->z_host, gpuSolver->z_in, _ni*_nj*_nk*sizeof(float));
    // ready for advection and compensation
    // init velocity buffer will be stored in gpu.du, gpu.dv, gpu.dw
    gpuSolver->copyHostToDevice(u_init, gpuSolver->u_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(v_init, gpuSolver->v_host, gpuSolver->dv, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(w_init, gpuSolver->w_host, gpuSolver->dw, _ni*_nj*(_nk+1)*sizeof(float));
    // updated velocity(no compensation) will be store in gpu.u, gpu.v, gpu.w
    gpuSolver->advectVelocity(false);
    // compensated and clamped velocity will be stored in gpu.u, gpu.v, gpu.w
    gpuSolver->compensateVelocity(false);
    // now copy backward_prev to gpu.x_out, gpu._yout, gpu.z_out
    gpuSolver->copyHostToDevice(backward_xprev, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_yprev, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_zprev, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(u_prev, gpuSolver->u_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(v_prev, gpuSolver->v_host, gpuSolver->dv, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(w_prev, gpuSolver->w_host, gpuSolver->dw, _ni*_nj*(_nk+1)*sizeof(float));
    // velocity from backward_prev will be blended with velocity from backward_curr
    // if reinitialization is not happen, prev buffers are not valid
    if (total_reinit_count != 0)
        gpuSolver->advectVelocityDouble(false, blend_coeff);
    else
        gpuSolver->advectVelocityDouble(false, 1.f);
    // copy velocity back to host
    gpuSolver->copyDeviceToHost(un, gpuSolver->u_host, gpuSolver->u);
    gpuSolver->copyDeviceToHost(vn, gpuSolver->v_host, gpuSolver->v);
    gpuSolver->copyDeviceToHost(wn, gpuSolver->w_host, gpuSolver->w);
}
```

And then, the solver calls the ''advectVelocity" function. The first step of this function is velocity advection. We look directly into the cuda code.

```
__global__ void advect_kernel(float *field, float *field_init,
                              float *backward_x, float *backward_y, float *backward_z,
                              float h, int ni, int nj, int nk,
                              int dimx, int dimy, int dimz, bool is_point)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (2+dimx<i && i<vel_buffer_i-3 && 2+dimy< j && j<vel_buffer_j-3 && 2+dimz<k && k<vel_buffer_k-3)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 pos = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                     float(j)*h + buffer_origin.y + volume[ii].y,
                                     float(k)*h + buffer_origin.z + volume[ii].z);

            float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
            float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
            float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

            float3 pos_init = make_float3(x_init, y_init, z_init);

            pos_init = clampv3(pos_init, make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
            sum += weight*sample_buffer(field_init, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, pos_init);
        }
        float3 pos = make_float3(float(i)*h + buffer_origin.x,
                                 float(j)*h + buffer_origin.y,
                                 float(k)*h + buffer_origin.z);

        float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
        float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
        float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

        float3 pos_init = make_float3(x_init, y_init, z_init);

        pos_init = clampv3(pos_init, make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
        float value = sample_buffer(field_init, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, pos_init);
        field[index] = 0.5f*sum + 0.5f*value;
    }
    __syncthreads();
}
```

This function is a standard velocity advection. Through backward mapping, we can get the initial position of the imaginary particle. We use it to interpolate in the u_init,v_init,w_init field, and the result is stored in GPUsolver.u,v,w. We can notice that it combines the samples from both grid center and other positions in the grid.

```
__global__ void compensate_kernel(float *src_buffer, float *temp_buffer, float *test_buffer,
                                  float *x_map, float *y_map, float *z_map,
                                  float h, int ni, int nj, int nk,
                                  int dimx, int dimy, int dimz, bool is_point)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (1+dimx<i && i<vel_buffer_i-2 && 1+dimy< j && j<vel_buffer_j-2 && 1+dimz<k && k<vel_buffer_k-2)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 point = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                       float(j)*h + buffer_origin.y + volume[ii].y,
                                       float(k)*h + buffer_origin.z + volume[ii].z);
            float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float3 map_pos = make_float3(x_pos, y_pos, z_pos);
            map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
            sum += weight * sample_buffer(src_buffer, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        }
        float3 point = make_float3(float(i)*h + buffer_origin.x,
                                   float(j)*h + buffer_origin.y,
                                   float(k)*h + buffer_origin.z);
        // forward mapping position
        float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float3 map_pos = make_float3(x_pos, y_pos, z_pos);
        map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
        float value = sample_buffer(src_buffer, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        sum = 0.5*sum + 0.5*value;
        test_buffer[index] = sum - temp_buffer[index];
//        sum -= temp_buffer[index];
//        sum *= 0.5f;
//        temp_buffer[index] = sum;
    }
    __syncthreads();
}
```

Next, we come to the compensate kernel. This function does forward mapping to get the mapping position. Then it uses this position to interpolate in the velocity field from the velocity advection that we just completed. Finally, this interpolation result is subtracted by the initial velocity field(u_init,v_init,w_init) and stored in the return value(u_src,v_src,w_src).

We can notice that this two functions above is just the implementation of BFECC error correction. (I assume you have some knowledge of it) And the error is stored in  u_src,v_src,w_src.

<img src="/images/img3.png" alt="img1" width="50%">

So the final step should be accumulating the error, and this is what "cumulate_kernel" is doing.

```
__global__ void cumulate_kernel(float *dfield, float *dfield_init,
                                float *x_map, float *y_map, float *z_map,
                                float h, int ni, int nj, int nk,
                                int dimx, int dimy, int dimz, bool is_point, float coeff)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (1+dimx<i && i<vel_buffer_i-2 && 1+dimy< j && j<vel_buffer_j-2 && 1+dimz<k && k<vel_buffer_k-2)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 point = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                       float(j)*h + buffer_origin.y + volume[ii].y,
                                       float(k)*h + buffer_origin.z + volume[ii].z);
            // forward mapping position
            // also used in compensation
            float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
            float3 map_pos = make_float3(x_pos, y_pos, z_pos);
            map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
            sum += weight * coeff * sample_buffer(dfield, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        }
        float3 point = make_float3(float(i)*h + buffer_origin.x,
                                   float(j)*h + buffer_origin.y,
                                   float(k)*h + buffer_origin.z);
        // forward mapping position
        float x_pos = sample_buffer(x_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float y_pos = sample_buffer(y_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float z_pos = sample_buffer(z_map,ni,nj,nk,h,make_float3(0.0,0.0,0.0),point);
        float3 map_pos = make_float3(x_pos, y_pos, z_pos);
        map_pos = clampv3(map_pos, make_float3(0,0,0), make_float3(h*float(ni), h*float(nj), h*float(nk)));
        float value = coeff * sample_buffer(dfield, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, map_pos);
        sum = 0.5*sum + 0.5 * value;
        dfield_init[index] += sum;
    }
    __syncthreads();
}
```

It is worth mentioning that the error we computed is in the "initial space",which means that the imaginary particles corresponding to the error grid are in the position of initial time. You can easily figure out the reason from the formula listed above. So we need to do some backward mapping before accumulating the error to the current velocity. 

```
__global__ void doubleAdvect_kernel(float *field, float *temp_field,
                                    float *backward_x, float *backward_y, float * backward_z,
                                    float *backward_xprev, float *backward_yprev, float *backward_zprev,
                                    float h, int ni, int nj, int nk,
                                    int dimx, int dimy, int dimz, bool is_point, float blend_coeff)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);


    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }


    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);
    if (2+dimx<i && i<vel_buffer_i-3 && 2+dimy< j && j<vel_buffer_j-3 && 2+dimz<k && k<vel_buffer_k-3)
    {
        float sum = 0.0;
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 pos = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                     float(j)*h + buffer_origin.y + volume[ii].y,
                                     float(k)*h + buffer_origin.z + volume[ii].z);
            float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
            float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
            float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

            float3 midpos = make_float3(x_init, y_init, z_init);
            midpos = clampv3(midpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) -h ));
            float x_orig = sample_buffer(backward_xprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
            float y_orig = sample_buffer(backward_yprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
            float z_orig = sample_buffer(backward_zprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
            float3 finalpos = make_float3(x_orig, y_orig, z_orig);

            finalpos = clampv3(finalpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
            sum += weight*sample_buffer(temp_field, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, finalpos);
        }
        float3 pos = make_float3(float(i)*h + buffer_origin.x,
                                 float(j)*h + buffer_origin.y,
                                 float(k)*h + buffer_origin.z);
        float x_init = sample_buffer(backward_x, ni, nj, nk, h, make_float3(0,0,0), pos);
        float y_init = sample_buffer(backward_y, ni, nj, nk, h, make_float3(0,0,0), pos);
        float z_init = sample_buffer(backward_z, ni, nj, nk, h, make_float3(0,0,0), pos);

        float3 midpos = make_float3(x_init, y_init, z_init);
        midpos = clampv3(midpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) -h ));
        float x_orig = sample_buffer(backward_xprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
        float y_orig = sample_buffer(backward_yprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
        float z_orig = sample_buffer(backward_zprev, ni, nj, nk, h, make_float3(0,0,0), midpos);
        float3 finalpos = make_float3(x_orig, y_orig, z_orig);

        finalpos = clampv3(finalpos,make_float3(h,h,h), make_float3(h*float(ni) - h, h*float(nj) - h, h*float(nk) - h));
        float value = sample_buffer(temp_field, ni+dimx, nj+dimy, nk+dimz, h, buffer_origin, finalpos);
        float prev_value = 0.5f*(sum + value);
        field[index] = field[index]*blend_coeff + (1-blend_coeff)*prev_value;
    }
    __syncthreads();
}
```

The last part of velocity advection is double advection. The reason we need this part is that since the program does re-initialization from time to time, the "initial velocity" is not the velocity when time equals to 0, but the velocity when the latest re-initialization happens. So that we need to introduce the mapping before the latest re-initialization, which is backward prev in the code, to avoid drastic state change when re-initialization happens. Similarly, the uprev,vprev and wprev in the code is the velocity when the second-to-last initialization occurs. I want to remind you that uinit and uprev include another term apart from initial velocity, and I choose to save it for later discussion.

Until now, the advection part is finished.



##  accumulated fluid change

```
	_utemp.copy(_un);
    _vtemp.copy(_vn);
    _wtemp.copy(_wn);
    _rhotemp.copy(_rho);
    _Ttemp.copy(_T);

    clearBoundary(_rho);
    emitSmoke(framenum, dt);
    addBuoyancy(dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(dt, viscosity, _un);
        diffuse_field(dt, viscosity, _vn);
        diffuse_field(dt, viscosity, _wn);
    }

    // calculate velocity change due to external forces(e.g. buoyancy)
    _duextern.copy(_un); _duextern -= _utemp;
    _dvextern.copy(_vn); _dvextern -= _vtemp;
    _dwextern.copy(_wn); _dwextern -= _wtemp;

    _utemp.copy(_un);
    _vtemp.copy(_vn);
    _wtemp.copy(_wn);
    projection();
    // calculate velocity change due to pressure projection
    _duproj.copy(_un); _duproj -= _utemp;
    _dvproj.copy(_vn); _dvproj -= _vtemp;
    _dwproj.copy(_wn); _dwproj -= _wtemp;
    _drhoextern.copy(_rho); _drhoextern -= _rhotemp;
    _dTextern.copy(_T); _dTextern -= _Ttemp;
```

We continue to calculate other components of discrete NS equation. 

After finish external forces calculation and pressure projection, we need to store the velocity change, which will be useful later.

```
	float VelocityDistortion = VelocityAdvector.estimateDistortion(_b_desc) / (max_v * dt);
    float ScalarDistortion = ScalarAdvector.estimateDistortion(_b_desc) / (max_v * dt);
    cout << "[ Velocity Distortion is " << VelocityDistortion << " ]" << endl;
    cout << "[ Scalar Distortion is " << ScalarDistortion << " ]" << endl;
```

As mentioned in the paper, we need to calculate distortion to decide whether we will do re-initialization.

```
	VelocityAdvector.accumulateVelocity(_uinit, _vinit, _winit, _duextern, _dvextern, _dwextern, 1.f);
    VelocityAdvector.accumulateVelocity(_uinit, _vinit, _winit, _duproj, _dvproj, _dwproj, proj_coeff);
    ScalarAdvector.accumulateField(_rhoinit, _drhoextern);
    ScalarAdvector.accumulateField(_Tinit, _dTextern);
```

The velocity change computed above comes in handy here, we accumulate the velocity change to uinit,vinit,winit. In contrast to the accumulation done in velocity compensation, we map it to the initial time by sampling it with the forward mapping. Because the velocity change is in "current space", and the initial velocity is in "initial space".

The accumulation of velocity change:

<img src="/images/img4.png" alt="img1" width="60%">

Therefore, the true meaning of uinit,vinit,winit(also uprev,vprev and wprev) is revealed. It is the sum of initial velocity field and the velocity change field converted to  "initial space". Thus, we can extend the BFECC formula to the formula below:

<img src="/images/img5.png" alt="img1" width="60%">



## re-initialization

```
void MapperBase::reinitializeMapping()
{
    total_reinit_count ++;
    backward_xprev.copy(backward_x);
    backward_yprev.copy(backward_y);
    backward_zprev.copy(backward_z);

    int compute_elements = forward_x._blockx*forward_x._blocky*forward_x._blockz;
    int slice = forward_x._blockx*forward_x._blocky;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/forward_x._blockx;
        uint bi = thread_idx%(forward_x._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(i<forward_x._nx && j<forward_x._ny && k<forward_x._nz)
            {
                float world_x = ((float)i-forward_x._ox)*_h;
                float world_y = ((float)j-forward_x._oy)*_h;
                float world_z = ((float)k-forward_x._oz)*_h;
                forward_x(i,j,k) = world_x;
                forward_y(i,j,k) = world_y;
                forward_z(i,j,k) = world_z;
                backward_x(i,j,k) = world_x;
                backward_y(i,j,k) = world_y;
                backward_z(i,j,k) = world_z;
            }
        }
    });
}
```

Finally, if the error exceeds the threshold,  we need to do re-initialization. We copy the backward to the backward prev, and set both forward and backward mapping to invariant mapping.

```
void BimocqSolver::velocityReinitialize()
{
    _uprev.copy(_uinit);
    _vprev.copy(_vinit);
    _wprev.copy(_winit);
    // set current buffer as next initial buffer
    _uinit.copy(_un);
    _vinit.copy(_vn);
    _winit.copy(_wn);
}
```

The same operation is executed to initial velocity.



So that is all about the main component of Bimocq. However, there is still some details unsolved. For example, the boundary condition, the smoke emission and the pressure projection. I will discuss them in the essay coming soon.

