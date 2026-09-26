#pragma once

struct Observation {
    double x = 0.0;                  // wala on the unit square
    double y = 0.0;                  // wac on the unit square
    double z = 0.0;                  // the third axis on the unit interval (three-dimensional meshes; DMESH_AXIS3)
    double working_weight = 0.0;     // binomial information at the baseline
    double working_response = 0.0;   // working response at the baseline (logits)
    int events = 0;
    int exposure = 0;
    int pool_id = 0;
    double offset = 0.0;             // the row's baseline logit: the pooled level plus any offset column
    double attribute_offset = 0.0;   // the offset column, before the pooled level is added
};
