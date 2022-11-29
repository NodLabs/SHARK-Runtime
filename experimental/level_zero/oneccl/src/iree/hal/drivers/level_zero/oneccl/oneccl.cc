// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "oneccl.h"

#include <experimental/level_zero/direct_command_buffer.h>
#include <experimental/level_zero/level_zero_buffer.h>
#include <experimental/level_zero/level_zero_driver.h>
#include <experimental/level_zero/status_util.h>
#include <mpi.h>
#include <ze_api.h>

#include <cstdlib>
#include <exception>
#include <memory>
#include <oneapi/ccl.hpp>
#include <sstream>
#include <utility>

static iree_hal_channel_vtable_t
iree_hal_level_zero_oneccl_channel_create_vtable();

struct iree_hal_level_zero_oneccl_device_t {
  iree_hal_level_zero_oneccl_device_t(ccl::context&& context,
                                      ccl::device&& device,
                                      ccl::stream&& stream)
      : context(std::move(context)),
        device(std::move(device)),
        stream(std::move(stream)) {}
  ccl::context context;
  ccl::device device;
  ccl::stream stream;
};

namespace {

iree_hal_channel_vtable_t iree_hal_level_zero_oneccl_channel_vtable =
    iree_hal_level_zero_oneccl_channel_create_vtable();

struct iree_hal_level_zero_oneccl_channel_t {
  iree_hal_level_zero_oneccl_channel_t(ccl::communicator&& communicator)
      : communicator(std::move(communicator)) {
    iree_hal_resource_initialize(&iree_hal_level_zero_oneccl_channel_vtable,
                                 &resource);
  }
  iree_hal_resource_t resource;
  ccl::communicator communicator;
};

iree_hal_level_zero_oneccl_channel_t* iree_hal_level_zero_oneccl_channel_cast(
    iree_hal_channel_t* channel) {
  IREE_HAL_ASSERT_TYPE(channel, &iree_hal_level_zero_oneccl_channel_vtable);
  return reinterpret_cast<iree_hal_level_zero_oneccl_channel_t*>(channel);
}

const iree_hal_level_zero_oneccl_channel_t*
iree_hal_level_zero_oneccl_channel_cast(const iree_hal_channel_t* channel) {
  IREE_HAL_ASSERT_TYPE(channel, &iree_hal_level_zero_oneccl_channel_vtable);
  return reinterpret_cast<const iree_hal_level_zero_oneccl_channel_t*>(channel);
}

iree_status_t iree_hal_collective_element_type_to_oneccl_datatype(
    iree_hal_collective_element_type_t element_type, ccl::datatype* res) {
  switch (element_type) {
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_8:
      *res = ccl::datatype::int8;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_8:
      *res = ccl::datatype::uint8;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_16:
      *res = ccl::datatype::int16;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_16:
      *res = ccl::datatype::uint16;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_32:
      *res = ccl::datatype::int32;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_32:
      *res = ccl::datatype::uint32;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_64:
      *res = ccl::datatype::int64;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_64:
      *res = ccl::datatype::uint64;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_16:
      *res = ccl::datatype::float16;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_32:
      *res = ccl::datatype::float32;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_64:
      *res = ccl::datatype::float64;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_BFLOAT_16:
      *res = ccl::datatype::bfloat16;
      break;
    default:
      return iree_make_status_with_location(
          __FILE__, __LINE__, IREE_STATUS_UNKNOWN,
          "Unknown iree_hal_collective_element_type_t %d encountered when "
          "converting to ccl::datatype.",
          element_type);
  }
  return iree_ok_status();
}

}  // namespace

void iree_hal_level_zero_oneccl_device_destroy(
    iree_hal_level_zero_oneccl_device_t* d) {
  delete d;
}

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
  static const char* error_msg = "Failed to create oneCCL channel.";
  iree_hal_level_zero_device_t* hal_level_zero_device =
      iree_hal_level_zero_device_cast(base_device);
  iree_hal_level_zero_driver_t* hal_level_zero_driver =
      iree_hal_level_zero_driver_cast(hal_level_zero_device->driver);

  try {
    if (!hal_level_zero_device->oneccl_device) {
      // Although the free function |cl::sycl::make_device|
      // should not need the creation of a sycl_platform, without it you get:
      // terminate called after throwing an instance of
      // 'cl::sycl::runtime_error' what():  Native API failed. Native API
      // returns: -30 (CL_INVALID_VALUE) -30 (CL_INVALID_VALUE)
      // https://github.com/intel/llvm/issues/5769
      sycl::platform sycl_platform =
          sycl::make_platform<sycl::backend::ext_oneapi_level_zero>(
              hal_level_zero_driver->driver_handle);

      cl::sycl::device sycl_device =
          cl::sycl::make_device<cl::sycl::backend::ext_oneapi_level_zero>(
              cl::sycl::backend_input_t<
                  cl::sycl::backend::ext_oneapi_level_zero, cl::sycl::device>(
                  hal_level_zero_device->device));
      ccl::device ccl_device = ccl::create_device(sycl_device);
      cl::sycl::context sycl_context(sycl_device);
      ccl::context ccl_context = ccl::create_context(sycl_context);
      cl::sycl::queue sycl_queue =
          cl::sycl::make_queue<cl::sycl::backend::ext_oneapi_level_zero>(
              cl::sycl::backend_input_t<
                  cl::sycl::backend::ext_oneapi_level_zero, cl::sycl::queue>(
                  hal_level_zero_device->command_queue, sycl_device,
                  cl::sycl::ext::oneapi::level_zero::ownership::keep),
              sycl_context);
      ccl::stream ccl_stream = ccl::create_stream(sycl_queue);
      std::unique_ptr<iree_hal_level_zero_oneccl_device_t> iree_oneccl_device =
          std::make_unique<iree_hal_level_zero_oneccl_device_t>(
              std::move(ccl_context), std::move(ccl_device),
              std::move(ccl_stream));
      hal_level_zero_device->oneccl_device = iree_oneccl_device.release();
    }

    // TODO: figure out a way to initialize MPI externally and pass a
    // communicator as an intialization parameter to oneCCL.
    int is_mpi_initialized;
    MPI_Initialized(&is_mpi_initialized);
    if (!is_mpi_initialized) {
      MPI_Init(nullptr, nullptr);
    }

    int rank;
    if (params.rank == IREE_HAL_CHANNEL_RANK_DEFAULT) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    } else {
      rank = params.rank;
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
      kvs = ccl::create_main_kvs();
      main_addr = kvs->get_address();
      MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
                MPI_COMM_WORLD);
    } else {
      MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
                MPI_COMM_WORLD);
      kvs = ccl::create_kvs(main_addr);
    }

    ccl::communicator communicator = ccl::create_communicator(
        size, rank, hal_level_zero_device->oneccl_device->device,
        hal_level_zero_device->oneccl_device->context, kvs);

    std::unique_ptr<iree_hal_level_zero_oneccl_channel_t> channel =
        std::make_unique<iree_hal_level_zero_oneccl_channel_t>(
            std::move(communicator));
    *out_channel = reinterpret_cast<iree_hal_channel_t*>(channel.release());
  } catch (std::exception& e) {
    std::stringstream msg;
    msg << error_msg << " " << e.what();
    return iree_make_status(IREE_STATUS_UNKNOWN, msg.str().c_str());
  } catch (...) {
    return iree_make_status(IREE_STATUS_UNKNOWN, error_msg);
  }

  return iree_ok_status();
}

void iree_hal_level_zero_oneccl_channel_destroy(iree_hal_channel_t* channel) {
  iree_hal_level_zero_oneccl_channel_t* oneccl_channel =
      iree_hal_level_zero_oneccl_channel_cast(channel);
  delete oneccl_channel;
}

void iree_hal_level_zero_oneccl_channel_query_rank_and_count(
    const iree_hal_channel_t* channel, int32_t* out_rank, int32_t* out_count) {
  const iree_hal_level_zero_oneccl_channel_t* oneccl_channel =
      iree_hal_level_zero_oneccl_channel_cast(channel);
  *out_count = oneccl_channel->communicator.size();
  *out_rank = oneccl_channel->communicator.rank();
}

// iree_status_t make_sycl_device(ze_device_handle_t level_zero_device,
//                                iree_hal_level_zero_dynamic_symbols_t* syms,
//                                cl::sycl::device* out_device) {
//   ze_device_properties_t ze_device_properties;
//   LEVEL_ZERO_RETURN_IF_ERROR(
//       syms, zeDeviceGetProperties(level_zero_device, &ze_device_properties),
//       "Failure at zeDeviceGetProperties.");
//   static const size_t uuid_size = 16;
//   if (sizeof(ze_device_properties.uuid.id) != uuid_size) {
//     return iree_make_status(IREE_STATUS_NOT_FOUND,
//                             "Unexpected Level Zero device UUID size.");
//   }
//   try {
//     std::vector<cl::sycl::device> sycl_devices =
//         cl::sycl::device::get_devices();
//     for (auto& sycl_device : sycl_devices) {
//       std::array<unsigned char, 16> sycl_device_uuid =
//           std::array<unsigned char, 16>(
//               sycl_device.get_info<
//                   cl::sycl::info::device::ext_intel_device_info_uuid>());
//       cl::sycl::backend sycl_device_backend = sycl_device.get_backend();
//       if (memcmp(sycl_device_uuid.data(), ze_device_properties.uuid.id,
//                  sycl_device_uuid.size()) == 0 &&
//           sycl_device_backend == cl::sycl::backend::ext_oneapi_level_zero) {
//         *out_device = sycl_device;
//         return iree_ok_status();
//       }
//     }
//   } catch (...) {
//   }
//   return iree_make_status(
//       IREE_STATUS_UNKNOWN,
//       "Failed to create oneCCL device that matches a Level Zero device.");
// }

iree_status_t iree_hal_level_zero_direct_command_buffer_oneccl_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  iree_hal_level_zero_direct_command_buffer_t* hal_ze_cmd_buff =
      iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);
  iree_hal_level_zero_oneccl_channel_t* oneccl_channel =
      iree_hal_level_zero_oneccl_channel_cast(channel);

  ccl::datatype oneccl_dtype;
  IREE_RETURN_IF_ERROR(iree_hal_collective_element_type_to_oneccl_datatype(
      op.element_type, &oneccl_dtype));
  // size_t dtype_size = ccl::get_datatype_size(oneccl_dtype);

  iree_hal_level_zero_device_ptr_t device_send_buff =
      iree_hal_level_zero_buffer_binding_device_pointer(send_binding);
  iree_hal_level_zero_device_ptr_t device_recv_buff =
      iree_hal_level_zero_buffer_binding_device_pointer(recv_binding);

  // // Compute element count.
  // size_t element_count = 0;
  // switch (op.kind) {
  //   case IREE_HAL_COLLECTIVE_KIND_ALL_GATHER:
  //     auto len_div = std::div(static_cast<size_t>(send_binding.length),
  //     dtype_size); element_count = len_div.quot; if (len_div.rem != 0) {
  //         return iree_make_status_with_location(__FILE__, __LINE__,
  //         IREE_STATUS_UNKNOWN,
  //           "Buffer length %d is not a multiple of element size %d.",
  //           send_binding.length, dtype_size);
  //     }
  //     break;
  //   default:
  //     return iree_make_status_with_location(__FILE__, __LINE__,
  //     IREE_STATUS_UNKNOWN,
  //       "Unsupported iree_hal_collective_kind_t %d.", op.kind);
  // }

  hal_ze_cmd_buff->command_segments->segments.push_back(
      [op, oneccl_dtype, element_count, device_send_buff, device_recv_buff,
       hal_ze_cmd_buff, oneccl_channel]() {
        std::vector<size_t> recv_counts(oneccl_channel->communicator.size(),
                                        element_count);
        switch (op.kind) {
          case IREE_HAL_COLLECTIVE_KIND_ALL_GATHER:
            ccl::allgatherv(device_send_buff, element_count, device_recv_buff,
                            recv_counts, oneccl_dtype,
                            oneccl_channel->communicator,
                            hal_ze_cmd_buff->device->oneccl_device->stream);
            break;
          default:
            return iree_make_status_with_location(
                __FILE__, __LINE__, IREE_STATUS_UNKNOWN,
                "Unsupported iree_hal_collective_kind_t %d.", op.kind);
        }

        return iree_ok_status();
      });

  return iree_ok_status();
}

static iree_hal_channel_vtable_t
iree_hal_level_zero_oneccl_channel_create_vtable() {
  return {.destroy = iree_hal_level_zero_oneccl_channel_destroy,
          .query_rank_and_count =
              iree_hal_level_zero_oneccl_channel_query_rank_and_count};
}
