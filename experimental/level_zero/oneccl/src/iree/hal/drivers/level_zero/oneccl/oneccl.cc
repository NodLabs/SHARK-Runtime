#include "oneccl.h"

#include <ze_api.h>

#include <oneapi/ccl.hpp>

extern "C" {

void iree_hal_level_zero_device_oneccl_init_vtable(
    iree_hal_device_vtable_t* vtable) {
  vtable->create_channel = iree_hal_level_zero_device_oneccl_create_channel;
}

void iree_hal_level_zero_direct_command_buffer_oneccl_init_vtable(
    iree_hal_command_buffer_vtable_t* vtable) {
  vtable->collective =
      iree_hal_level_zero_direct_command_buffer_oneccl_collective;
}

iree_status_t iree_hal_level_zero_device_oneccl_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

iree_status_t iree_hal_level_zero_direct_command_buffer_oneccl_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}
}
