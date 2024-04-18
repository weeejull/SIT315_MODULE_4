__kernel void vector_add_ocl(const int size, __global int *v1, __global int *v2, __global int *v_out) {
    const int globalIndex = get_global_id(0); // GET THE GLOBAL INDEX OF THE CURRENT WORK-ITEM
    
    if (globalIndex < size) { // CHECK IF THE GLOBAL INDEX IS WITHIN THE ARRAY SIZE
        // PERFORM VECTOR ADDITION: ADD CORRESPONDING ELEMENTS FROM v1 AND v2 ARRAYS AND STORE THE RESULT IN v_out ARRAY
        v_out[globalIndex] = v1[globalIndex] + v2[globalIndex];
    }
}

