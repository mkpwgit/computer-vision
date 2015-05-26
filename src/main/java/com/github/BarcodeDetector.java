package com.github;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

class BarcodeDetector {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        String fileName = "F:\\barcode5.jpg";

        //load image
        Mat image = Highgui.imread(fileName);
        Mat grayImage = Highgui.imread(fileName);
        Mat gradX = Highgui.imread(fileName);
        Mat gradY = Highgui.imread(fileName);
        Mat gradient = Highgui.imread(fileName);
        Mat gradientResult = Highgui.imread(fileName);
        Mat blurImage = Highgui.imread(fileName);
        Mat thresholdImage = Highgui.imread(fileName);
        Mat closedImage = Highgui.imread(fileName);
        Mat closedImage2 = Highgui.imread(fileName);
        Mat closedImage3 = Highgui.imread(fileName);
        Mat hierarchy = Highgui.imread(fileName);

        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);

        Imgproc.Sobel(grayImage, gradX, CvType.CV_32F, 1, 0);
        Imgproc.Sobel(gradX, gradY, CvType.CV_32F, 0, 1);

        Core.subtract(gradX, gradY, gradient);
        Core.convertScaleAbs(gradient, gradientResult);

        Imgproc.blur(gradientResult, blurImage, new Size(9.0, 9.0));

        Imgproc.threshold(blurImage, thresholdImage, 155.0, 155.0, Imgproc.THRESH_BINARY);

        Mat kernelImage = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(21.0, 7.0));
        Imgproc.morphologyEx(thresholdImage, closedImage, Imgproc.MORPH_CLOSE, kernelImage);

        Imgproc.erode(closedImage, closedImage2, kernelImage);
        Imgproc.dilate(closedImage2, closedImage3, kernelImage);

        List<MatOfPoint> matOfPoints = new ArrayList<>();
        //check null!!!
        Imgproc.findContours(closedImage3, matOfPoints, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Point> points = new ArrayList<>();
        int maxSize = 0;
        int number = 0;
        for (int i = 0; i < matOfPoints.size(); i++) {
            List<Point> currentPoints = matOfPoints.get(i).toList();
            if (currentPoints.size() > maxSize) {
                points = currentPoints;
                number = i;
            }

        }

        List<MatOfPoint> resultList = new ArrayList<>();
        resultList.add(matOfPoints.get(number));

        MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
        matOfPoint2f.fromList(points);

        RotatedRect rect = Imgproc.minAreaRect(matOfPoint2f);


        Point points2[] = new Point[4];
        rect.points(points2);
        for (int i = 0; i < 4; ++i) {
            Core.line(image, points2[i], points2[(i + 1) % 4], new Scalar(0, 255, 0), 3);
        }
//        Imgproc.drawContours(image, resultList, -1, new Scalar(0, 255, 0), 3);

        Highgui.imwrite("F:\\barcodeResult.jpg", image);


        System.out.println();

    }

}
