import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from pcdet.utils import box_utils

## This function is used to determine whether a point is inside a rectangle or not
class compute_vertex(Function):
    '''
    Compute the corners which are inside the rectangles
    '''

    @staticmethod
    def forward(ctx, corners_gboxes, corners_qboxes):

        np_corners_gboxes = corners_gboxes.cpu().numpy()
        np_corners_qboxes = corners_qboxes.cpu().detach().numpy()
        N = corners_gboxes.shape[0]
        num_of_intersections = np.zeros((N,), dtype=np.int32)
        intersections = np.zeros((N, 16), dtype=np.float32)
        flags_qboxes = np.zeros((N, 4), dtype=np.float32)
        flags_gboxes = np.zeros((N, 4), dtype=np.float32)
        flags_inters = np.zeros((N, 4, 4), dtype=np.float32)

        for iter in range(N):
            # step 1: determine how many corners from corners_gboxes inside the np_qboxes
            ab0 = np_corners_qboxes[iter, 2] - np_corners_qboxes[iter, 0]
            ab1 = np_corners_qboxes[iter, 3] - np_corners_qboxes[iter, 1]
            ad0 = np_corners_qboxes[iter, 6] - np_corners_qboxes[iter, 0]
            ad1 = np_corners_qboxes[iter, 7] - np_corners_qboxes[iter, 1]
            # print(f"# {ab0} {ab1} {ad0} {ad1}")
            for i in range(4):
                ap0 = np_corners_gboxes[iter, i * 2] - np_corners_qboxes[iter, 0]
                ap1 = np_corners_gboxes[iter, i * 2 + 1] - np_corners_qboxes[iter, 1]
                abab = ab0 * ab0 + ab1 * ab1
                abap = ab0 * ap0 + ab1 * ap1
                adad = ad0 * ad0 + ad1 * ad1
                adap = ad0 * ap0 + ad1 * ap1

                # debug
                # if abab<0.01 or adad<0.01:
                # print(f"# abab adad abnormal {ap0} {ap1} {abab} {abap} {adad} {adap}")
                # print(f"np_corners_qboxes iter = {np_corners_qboxes[iter]}")
                # print(f"np_corners_gboxes iter = {np_corners_gboxes[iter]}")

                # import pdb;pdb.set_trace()
                if (abab >= abap and abap >= 0 and adad >= adap and adap >= 0 and adad>0 and abab>0):
                    intersections[iter, num_of_intersections[iter] * 2] = np_corners_gboxes[iter, i * 2]
                    intersections[iter, num_of_intersections[iter] * 2 + 1] = np_corners_gboxes[iter, i * 2 + 1]
                    num_of_intersections[iter] += 1
                    flags_gboxes[iter, i] = 1.0

            # step 2: determine how many corners from np_qboxes inside corners_gboxes
            ab0 = np_corners_gboxes[iter, 2] - np_corners_gboxes[iter, 0]
            ab1 = np_corners_gboxes[iter, 3] - np_corners_gboxes[iter, 1]
            ad0 = np_corners_gboxes[iter, 6] - np_corners_gboxes[iter, 0]
            ad1 = np_corners_gboxes[iter, 7] - np_corners_gboxes[iter, 1]
            for i in range(4):
                ap0 = np_corners_qboxes[iter, i * 2] - np_corners_gboxes[iter, 0]
                ap1 = np_corners_qboxes[iter, i * 2 + 1] - np_corners_gboxes[iter, 1]
                abab = ab0 * ab0 + ab1 * ab1
                abap = ab0 * ap0 + ab1 * ap1
                adad = ad0 * ad0 + ad1 * ad1
                adap = ad0 * ap0 + ad1 * ap1
                if (abab >= abap and abap >= 0 and adad >= adap and adap >= 0):
                    intersections[iter, num_of_intersections[iter] * 2] = np_corners_qboxes[iter, i * 2]
                    intersections[iter, num_of_intersections[iter] * 2 + 1] = np_corners_qboxes[iter, i * 2 + 1]
                    num_of_intersections[iter] += 1
                    flags_qboxes[iter, i] = 1.0

            # step 3: find the intersection of all the edges
            for i in range(4):
                for j in range(4):
                    A = np.zeros((2,), dtype=np.float32)
                    B = np.zeros((2,), dtype=np.float32)
                    C = np.zeros((2,), dtype=np.float32)
                    D = np.zeros((2,), dtype=np.float32)

                    A[0] = np_corners_gboxes[iter, 2 * i]
                    A[1] = np_corners_gboxes[iter, 2 * i + 1]
                    B[0] = np_corners_gboxes[iter, 2 * ((i + 1) % 4)]
                    B[1] = np_corners_gboxes[iter, 2 * ((i + 1) % 4) + 1]

                    C[0] = np_corners_qboxes[iter, 2 * j]
                    C[1] = np_corners_qboxes[iter, 2 * j + 1]
                    D[0] = np_corners_qboxes[iter, 2 * ((j + 1) % 4)]
                    D[1] = np_corners_qboxes[iter, 2 * ((j + 1) % 4) + 1]

                    BA0 = B[0] - A[0]
                    BA1 = B[1] - A[1]
                    CA0 = C[0] - A[0]
                    CA1 = C[1] - A[1]
                    DA0 = D[0] - A[0]
                    DA1 = D[1] - A[1]

                    acd = DA1 * CA0 > CA1 * DA0
                    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
                    if acd != bcd:
                        abc = CA1 * BA0 > BA1 * CA0
                        abd = DA1 * BA0 > BA1 * DA0
                        if abc != abd:
                            DC0 = D[0] - C[0]
                            DC1 = D[1] - C[1]
                            ABBA = A[0] * B[1] - B[0] * A[1]
                            CDDC = C[0] * D[1] - D[0] * C[1]
                            DH = BA1 * DC0 - BA0 * DC1
                            Dx = ABBA * DC0 - BA0 * CDDC
                            Dy = ABBA * DC1 - BA1 * CDDC
                            # DH = (B[1] - A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (D[1] - C[1])
                            # Dx = (A[0] * B[1] - B[0] * A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (C[0] * D[1] - D[0] * C[1])
                            # Dy = (A[0] * B[1] - B[0] * A[1]) * (D[1] - C[1]) - (B[1] - A[1]) * (C[0] * D[1] - D[0] * C[1])
                            if (num_of_intersections[iter] > 7):
                                print("iter = ", iter)
                                print("(%.4f %.4f) (%.4f %.4f) (%.4f %.4f) (%.4f %.4f)" % (
                                    np_corners_gboxes[iter, 0], np_corners_gboxes[iter, 1],
                                    np_corners_gboxes[iter, 2], np_corners_gboxes[iter, 3],
                                    np_corners_gboxes[iter, 4], np_corners_gboxes[iter, 5],
                                    np_corners_gboxes[iter, 6], np_corners_gboxes[iter, 7]))
                                print("(%.4f %.4f) (%.4f %.4f) (%.4f %.4f) (%.4f %.4f)" % (
                                    np_corners_qboxes[iter, 0], np_corners_qboxes[iter, 1],
                                    np_corners_qboxes[iter, 2], np_corners_qboxes[iter, 3],
                                    np_corners_qboxes[iter, 4], np_corners_qboxes[iter, 5],
                                    np_corners_qboxes[iter, 6], np_corners_qboxes[iter, 7]))
                                continue
                            intersections[iter, num_of_intersections[iter] * 2] = Dx / DH
                            intersections[iter, num_of_intersections[iter] * 2 + 1] = Dy / DH
                            num_of_intersections[iter] += 1
                            flags_inters[iter, i, j] = 1.0

        ctx.save_for_backward(corners_qboxes)
        ctx.corners_gboxes = corners_gboxes
        ctx.flags_qboxes = flags_qboxes
        ctx.flags_gboxes = flags_gboxes
        ctx.flags_inters = flags_inters
        # conver numpy to tensor
        tensor_intersections = torch.from_numpy(intersections)
        tensor_num_of_intersections = torch.from_numpy(num_of_intersections)
        return tensor_intersections, tensor_num_of_intersections.detach()

    @staticmethod
    def backward(ctx, *grad_outputs):
        _variables = ctx.saved_tensors
        corners_qboxes = _variables[0]
        corners_gboxes = ctx.corners_gboxes
        flags_qboxes = ctx.flags_qboxes
        flags_gboxes = ctx.flags_gboxes
        flags_inters = ctx.flags_inters
        grad_output = grad_outputs[0]

        np_corners_gboxes = corners_gboxes.cpu().numpy()
        np_corners_qboxes = corners_qboxes.cpu().detach().numpy()

        N = flags_qboxes.shape[0]
        n_of_inter = np.zeros((N,), dtype=np.int32)

        ### Check whether here is correct or not
        Jacbian_qboxes = np.zeros((N, 8, 16), dtype=np.float32)
        Jacbian_gboxes = np.zeros((N, 8, 16), dtype=np.float32)

        for iter in range(N):

            for i in range(4):
                if (flags_gboxes[iter, i] > 0):
                    Jacbian_gboxes[iter, i * 2, n_of_inter[iter] * 2] += 1.0
                    Jacbian_gboxes[iter, i * 2 + 1, n_of_inter[iter] * 2 + 1] += 1.0
                    n_of_inter[iter] += 1

            for i in range(4):
                if (flags_qboxes[iter, i] > 0):
                    Jacbian_qboxes[iter, i * 2, n_of_inter[iter] * 2] += 1.0
                    Jacbian_qboxes[iter, i * 2 + 1, n_of_inter[iter] * 2 + 1] += 1.0
                    n_of_inter[iter] += 1

            for i in range(4):
                for j in range(4):
                    if (flags_inters[iter, i, j] > 0):
                        ###
                        A = np.zeros((2,), dtype=np.float32)
                        B = np.zeros((2,), dtype=np.float32)
                        C = np.zeros((2,), dtype=np.float32)
                        D = np.zeros((2,), dtype=np.float32)
                        A[0] = np_corners_gboxes[iter, 2 * i]
                        A[1] = np_corners_gboxes[iter, 2 * i + 1]

                        B[0] = np_corners_gboxes[iter, 2 * ((i + 1) % 4)]
                        B[1] = np_corners_gboxes[iter, 2 * ((i + 1) % 4) + 1]

                        C[0] = np_corners_qboxes[iter, 2 * j]
                        C[1] = np_corners_qboxes[iter, 2 * j + 1]

                        D[0] = np_corners_qboxes[iter, 2 * ((j + 1) % 4)]
                        D[1] = np_corners_qboxes[iter, 2 * ((j + 1) % 4) + 1]
                        BA0 = B[0] - A[0]
                        BA1 = B[1] - A[1]
                        CA0 = C[0] - A[0]
                        CA1 = C[1] - A[1]
                        DA0 = D[0] - A[0]
                        DA1 = D[1] - A[1]
                        acd = DA1 * CA0 > CA1 * DA0
                        bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])

                        if acd != bcd:
                            abc = CA1 * BA0 > BA1 * CA0
                            abd = DA1 * BA0 > BA1 * DA0
                            if abc != abd:
                                DC0 = D[0] - C[0]
                                DC1 = D[1] - C[1]
                                ABBA = A[0] * B[1] - B[0] * A[1]
                                CDDC = C[0] * D[1] - D[0] * C[1]
                                DH = BA1 * DC0 - BA0 * DC1
                                Dx = ABBA * DC0 - BA0 * CDDC
                                Dy = ABBA * DC1 - BA1 * CDDC

                                # DH = (B[1] - A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (D[1] - C[1])
                                # Dx = (A[0] * B[1] - B[0] * A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (C[0] * D[1] - D[0] * C[1])
                                det_DxA0 = B[1] * (D[0] - C[0]) + (C[0] * D[1] - D[0] * C[1])
                                det_DxA1 = - B[0] * (D[0] - C[0])
                                det_DxB0 = - A[1] * (D[0] - C[0]) - (C[0] * D[1] - D[0] * C[1])
                                det_DxB1 = A[0] * (D[0] - C[0])
                                det_DxC0 = - (A[0] * B[1] - B[0] * A[1]) - (B[0] - A[0]) * D[1]
                                det_DxC1 = (B[0] - A[0]) * D[0]
                                det_DxD0 = (A[0] * B[1] - B[0] * A[1]) + (B[0] - A[0]) * C[1]
                                det_DxD1 = -(B[0] - A[0]) * C[0]
                                # Dy = (A[0] * B[1] - B[0] * A[1]) * (D[1] - C[1]) - (B[1] - A[1]) * (C[0] * D[1] - D[0] * C[1])
                                det_DyA0 = B[1] * (D[1] - C[1])
                                det_DyA1 = - B[0] * (D[1] - C[1]) + (C[0] * D[1] - D[0] * C[1])
                                det_DyB0 = -  A[1] * (D[1] - C[1])
                                det_DyB1 = A[0] * (D[1] - C[1]) - (C[0] * D[1] - D[0] * C[1])

                                det_DyC0 = - (B[1] - A[1]) * D[1]
                                det_DyC1 = - (A[0] * B[1] - B[0] * A[1]) + (B[1] - A[1]) * D[0]
                                det_DyD0 = (B[1] - A[1]) * C[1]
                                det_DyD1 = (A[0] * B[1] - B[0] * A[1]) - (B[1] - A[1]) * C[0]
                                # DH = (B[1] - A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (D[1] - C[1])
                                det_DHA0 = (D[1] - C[1])
                                det_DHA1 = - (D[0] - C[0])
                                det_DHB0 = - (D[1] - C[1])
                                det_DHB1 = (D[0] - C[0])
                                det_DHC0 = - (B[1] - A[1])
                                det_DHC1 = (B[0] - A[0])
                                det_DHD0 = (B[1] - A[1])
                                det_DHD1 = - (B[0] - A[0])

                                DHDH = DH * DH
                                Jacbian_gboxes[iter, i * 2, n_of_inter[iter] * 2] += (det_DxA0 * DH - Dx * det_DHA0) / DHDH
                                Jacbian_gboxes[iter, i * 2, n_of_inter[iter] * 2 + 1] += (det_DyA0 * DH - Dy * det_DHA0) / DHDH

                                Jacbian_gboxes[iter, i * 2 + 1, n_of_inter[iter] * 2] += (det_DxA1 * DH - Dx * det_DHA1) / DHDH
                                Jacbian_gboxes[iter, i * 2 + 1, n_of_inter[iter] * 2 + 1] += (det_DyA1 * DH - Dy * det_DHA1) / DHDH

                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4), n_of_inter[iter] * 2] += (det_DxB0 * DH - Dx * det_DHB0) / DHDH
                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4), n_of_inter[iter] * 2 + 1] += (det_DyB0 * DH - Dy * det_DHB0) / DHDH

                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4) + 1, n_of_inter[iter] * 2] += (det_DxB1 * DH - Dx * det_DHB1) / DHDH
                                Jacbian_gboxes[iter, 2 * ((i + 1) % 4) + 1, n_of_inter[iter] * 2 + 1] += (det_DyB1 * DH - Dy * det_DHB1) / DHDH

                                Jacbian_qboxes[iter, j * 2, n_of_inter[iter] * 2] += (det_DxC0 * DH - Dx * det_DHC0) / DHDH
                                Jacbian_qboxes[iter, j * 2, n_of_inter[iter] * 2 + 1] += (det_DyC0 * DH - Dy * det_DHC0) / DHDH

                                Jacbian_qboxes[iter, j * 2 + 1, n_of_inter[iter] * 2] += (det_DxC1 * DH - Dx * det_DHC1) / DHDH
                                Jacbian_qboxes[iter, j * 2 + 1, n_of_inter[iter] * 2 + 1] += (det_DyC1 * DH - Dy * det_DHC1) / DHDH

                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4), n_of_inter[iter] * 2] += (det_DxD0 * DH - Dx * det_DHD0) / DHDH
                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4), n_of_inter[iter] * 2 + 1] += (det_DyD0 * DH - Dy * det_DHD0) / DHDH

                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4) + 1, n_of_inter[iter] * 2] += (det_DxD1 * DH - Dx * det_DHD1) / DHDH
                                Jacbian_qboxes[iter, 2 * ((j + 1) % 4) + 1, n_of_inter[iter] * 2 + 1] += (det_DyD1 * DH - Dy * det_DHD1) / DHDH

                                n_of_inter[iter] += 1

        tensor_Jacbian_gboxes = torch.from_numpy(Jacbian_gboxes).to(torch.device(corners_qboxes.device))
        tensor_Jacbian_qboxes = torch.from_numpy(Jacbian_qboxes).to(torch.device(corners_qboxes.device))
        grad_output_cuda = grad_output.to(torch.device(corners_qboxes.device))
        # print("grad_output_cuda =", grad_output_cuda.shape)
        tensor_grad_corners_gboxes = tensor_Jacbian_gboxes.matmul(grad_output_cuda.unsqueeze(2)).squeeze(2)
        tensor_grad_corners_qboxes = tensor_Jacbian_qboxes.matmul(grad_output_cuda.unsqueeze(2)).squeeze(2)
        return tensor_grad_corners_gboxes, tensor_grad_corners_qboxes

class sort_vertex(Function):
    @staticmethod
    def forward(ctx, int_pts, num_of_inter):
        np_int_pts = int_pts.detach().numpy()
        #np_num_of_inter = num_of_inter.detach().numpy()
        np_num_of_inter = num_of_inter
        N = int_pts.shape[0]
        np_sorted_indexs = np.zeros((N, 8), dtype=np.int32)
        sorted_int_pts = np.zeros((N, 16), dtype=np.float32)
        for iter in range(N):
            if np_num_of_inter[iter] > 0:
                center = np.zeros((2,), dtype=np.float32)
                for i in range(np_num_of_inter[iter]):
                    center[0] += np_int_pts[iter, 2 * i]
                    center[1] += np_int_pts[iter, 2 * i + 1]
                center[0] /= np_num_of_inter[iter].float()
                center[1] /= np_num_of_inter[iter].float()

                angle = np.zeros((8,), dtype=np.float32)
                v = np.zeros((2,), dtype=np.float32)

                for i in range(np_num_of_inter[iter]):
                    v[0] = np_int_pts[iter, 2 * i] - center[0]
                    v[1] = np_int_pts[iter, 2 * i + 1] - center[1]
                    d = math.sqrt(v[0] * v[0] + v[1] * v[1])
                    v[0] = v[0] / d
                    v[1] = v[1] / d
                    anglei = math.atan2(v[1], v[0])
                    if anglei < 0:
                        angle[i] = anglei + 2 * 3.1415926
                    else:
                        angle[i] = anglei
                # sort angles with descending
                np_sorted_indexs[iter, :] = np.argsort(-angle)
                for i in range(np_num_of_inter[iter]):
                    sorted_int_pts[iter, 2 * i] = np_int_pts[iter, 2 * np_sorted_indexs[iter, i]]
                    sorted_int_pts[iter, 2 * i + 1] = np_int_pts[iter, 2 * np_sorted_indexs[iter, i] + 1]

        # conver numpy to tensor
        ctx.save_for_backward(int_pts, num_of_inter)
        ctx.np_sorted_indexs = np_sorted_indexs
        tensor_sorted_int_pts = torch.from_numpy(sorted_int_pts)
        return tensor_sorted_int_pts

    @staticmethod
    def backward(ctx, grad_output):
        int_pts, num_of_inter = ctx.saved_tensors
        np_sorted_indexs = ctx.np_sorted_indexs

        N = int_pts.shape[0]
        Jacbian_int_pts = np.zeros((N, 16, 16), dtype=np.float32)
        for iter in range(N):
            for i in range(num_of_inter[iter]):
                Jacbian_int_pts[iter, 2 * np_sorted_indexs[iter, i], 2 * i] = 1
                Jacbian_int_pts[iter, 2 * np_sorted_indexs[iter, i] + 1, 2 * i + 1] = 1

        tensor_Jacbian_int_pts = torch.from_numpy(Jacbian_int_pts).to(torch.device(int_pts.device))
        grad_output_cuda = grad_output.to(torch.device(int_pts.device))
        tensor_grad_int_pts = tensor_Jacbian_int_pts.matmul(grad_output_cuda.unsqueeze(2)).squeeze(2)
        # todo: my second addtion
        # my_add_1 = torch.zeros(tensor_grad_int_pts.shape[0], dtype=torch.float32)
        return tensor_grad_int_pts, None


class area_polygon(Function):

    @staticmethod
    def forward(ctx, int_pts, num_of_inter):
        ctx.save_for_backward(int_pts, num_of_inter)
        np_int_pts = int_pts.detach().numpy()
        #np_num_of_inter = num_of_inter.detach().numpy()
        np_num_of_inter = num_of_inter
        N = int_pts.shape[0]
        areas = np.zeros((N,), dtype=np.float32)

        for iter in range(N):
            for i in range(np_num_of_inter[iter] - 2):
                p1 = np_int_pts[iter, 0:2]
                p2 = np_int_pts[iter, 2 * i + 2:2 * i + 4]
                p3 = np_int_pts[iter, 2 * i + 4:2 * i + 6]
                areas[iter] += abs(((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) / 2.0)

        tensor_areas = torch.from_numpy(areas)

        return tensor_areas

    @staticmethod
    def backward(ctx, *grad_outputs):

        int_pts, num_of_inter = ctx.saved_tensors
        np_int_pts = int_pts.detach().numpy()
        np_num_of_inter = num_of_inter.detach().numpy()
        grad_output0 = grad_outputs[0]
        N = int_pts.shape[0]
        grad_int_pts = np.zeros((N, 16), dtype=np.float32)

        for iter in range(N):
            if (np_num_of_inter[iter] > 2):
                for i in range(np_num_of_inter[iter]):
                    if i == 0:
                        for j in range(np_num_of_inter[iter] - 2):
                            p1 = np_int_pts[iter, 0:2]
                            p2 = np_int_pts[iter, 2 * j + 2:2 * j + 4]
                            p3 = np_int_pts[iter, 2 * j + 4:2 * j + 6]

                            if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                                grad_int_pts[iter, 0] += (p2[1] - p3[1]) * grad_output0[iter] * 0.5
                                grad_int_pts[iter, 1] += -(p2[0] - p3[0]) * grad_output0[iter] * 0.5
                            else:
                                grad_int_pts[iter, 0] += -(p2[1] - p3[1]) * grad_output0[iter] * 0.5
                                grad_int_pts[iter, 1] += (p2[0] - p3[0]) * grad_output0[iter] * 0.5

                    elif i == 1:
                        p1 = np_int_pts[iter, 0:2]
                        p2 = np_int_pts[iter, 2:4]
                        p3 = np_int_pts[iter, 4:6]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, 2] = -(p1[1] - p3[1]) * grad_output0[iter] * 0.5
                            grad_int_pts[iter, 3] = (p1[0] - p3[0]) * grad_output0[iter] * 0.5
                        else:
                            grad_int_pts[iter, 2] = (p1[1] - p3[1]) * grad_output0[iter] * 0.5
                            grad_int_pts[iter, 3] = -(p1[0] - p3[0]) * grad_output0[iter] * 0.5

                    elif i == np_num_of_inter[iter] - 1:

                        p1 = np_int_pts[iter, 2 * (np_num_of_inter[iter] - 2):2 * (np_num_of_inter[iter] - 1)]
                        p2 = np_int_pts[iter, 2 * (np_num_of_inter[iter] - 1):2 * (np_num_of_inter[iter])]
                        p3 = np_int_pts[iter, 0:2]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, 2 * (np_num_of_inter[iter] - 1)] = - (p1[1] - p3[1]) * grad_output0[
                                iter] * 0.5
                            grad_int_pts[iter, 2 * np_num_of_inter[iter] - 1] = (p1[0] - p3[0]) * grad_output0[
                                iter] * 0.5
                        else:
                            grad_int_pts[iter, 2 * (np_num_of_inter[iter] - 1)] = (p1[1] - p3[1]) * grad_output0[
                                iter] * 0.5
                            grad_int_pts[iter, 2 * np_num_of_inter[iter] - 1] = - (p1[0] - p3[0]) * grad_output0[
                                iter] * 0.5
                    else:
                        p1 = np_int_pts[iter, 0:2]
                        p2 = np_int_pts[iter, 2 * i - 2: 2 * i]
                        p3 = np_int_pts[iter, 2 * i: 2 * i + 2]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, i * 2] += (- (p2[1] - p3[1]) + (p1[1] - p3[1])) * grad_output0[
                                iter] * 0.5
                            grad_int_pts[iter, i * 2 + 1] += (- (p1[0] - p3[0]) + (p2[0] - p3[0])) * grad_output0[
                                iter] * 0.5
                        else:
                            grad_int_pts[iter, i * 2] += ((p2[1] - p3[1]) - (p1[1] - p3[1])) * grad_output0[iter] * 0.5
                            grad_int_pts[iter, i * 2 + 1] += ((p1[0] - p3[0]) - (p2[0] - p3[0])) * grad_output0[
                                iter] * 0.5

                        p1 = np_int_pts[iter, 0:2]
                        p2 = np_int_pts[iter, 2 * i: 2 * i + 2]
                        p3 = np_int_pts[iter, 2 * i + 2: 2 * i + 4]
                        if ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p1[1] - p3[1]) * (p2[0] - p3[0])) > 0:
                            grad_int_pts[iter, i * 2] += - (p1[1] - p3[1]) * grad_output0[iter] * 0.5
                            grad_int_pts[iter, i * 2 + 1] += (p1[0] - p3[0]) * grad_output0[iter] * 0.5
                        else:
                            grad_int_pts[iter, i * 2] += (p1[1] - p3[1]) * grad_output0[iter] * 0.5
                            grad_int_pts[iter, i * 2 + 1] += -(p1[0] - p3[0]) * grad_output0[iter] * 0.5

        tensor_grad_int_pts = torch.from_numpy(grad_int_pts)
        # todo: my first addition.
        # my_add_0 = torch.zeros(tensor_grad_int_pts.shape[0], dtype=torch.float32)
        #print("area_polygon backward")
        return tensor_grad_int_pts, None


class rbbox_to_corners(nn.Module):

    def _init_(self, rbbox):
        super(rbbox_to_corners, self)._init_()
        self.rbbox = rbbox
        return

    def forward(ctx, rbbox):
        '''
                    There is no rotation performed here. As axis are aligned.
                                          ^ [y]
                                     1 --------- 2
                                     /          /    --->
                                    0 -------- 3     [x]
                    Each node has the coordinate of [x, y]. Corresponding the order of input.

                    Output: [N, 8]
                            [x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3],
                            if ry > 0, then rotate clockwisely.

                '''

        assert rbbox.shape[1] == 5
        device = rbbox.device
        corners = torch.zeros((rbbox.shape[0], 8), dtype=torch.float32, device=device)
        dxcos = rbbox[:, 2].mul(torch.cos(rbbox[:, 4])) / 2.0
        dxsin = rbbox[:, 2].mul(torch.sin(rbbox[:, 4])) / 2.0
        dycos = rbbox[:, 3].mul(torch.cos(rbbox[:, 4])) / 2.0
        dysin = rbbox[:, 3].mul(torch.sin(rbbox[:, 4])) / 2.0
        corners[:, 0] = -dxcos - dysin + rbbox[:, 0]
        corners[:, 1] = dxsin - dycos + rbbox[:, 1]
        corners[:, 2] = -dxcos + dysin + rbbox[:, 0]
        corners[:, 3] = dxsin + dycos + rbbox[:, 1]

        corners[:, 4] = dxcos + dysin + rbbox[:, 0]
        corners[:, 5] = -dxsin + dycos + rbbox[:, 1]
        corners[:, 6] = dxcos - dysin + rbbox[:, 0]
        corners[:, 7] = -dxsin - dycos + rbbox[:, 1]
        return corners

class rinter_area_compute(nn.Module):

    def _init_(self, corners_gboxes, corners_qboxes):
        super(rinter_area_compute, self)._init_()
        self.corners_gboxes = corners_gboxes
        self.corners_qboxes = corners_qboxes
        return

    def forward(ctx, corners_gboxes, corners_qboxes):
        intersections, num_of_intersections = compute_vertex.apply(corners_gboxes, corners_qboxes)
        num_of_intersections = num_of_intersections.detach()
        sorted_int_pts = sort_vertex.apply(intersections, num_of_intersections)
        # x = sorted_int_pts.clone()
        # x[0, 4:6] = sorted_int_pts[0, 6:8]
        # x[0, 6:8] = sorted_int_pts[0, 4:6]
        inter_area = area_polygon.apply(sorted_int_pts, num_of_intersections)
        return inter_area