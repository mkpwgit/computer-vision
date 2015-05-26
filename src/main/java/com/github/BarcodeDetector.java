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

        //keep some intermediate results
        Mat grayImage = Highgui.imread(fileName);
        Mat gradX = Highgui.imread(fileName);
        Mat gradY = Highgui.imread(fileName);
        Mat gradient = Highgui.imread(fileName);
        Mat blurImage = Highgui.imread(fileName);
        Mat thresholdImage = Highgui.imread(fileName);
        Mat closedImage = Highgui.imread(fileName);
        Mat hierarchy = Highgui.imread(fileName);

        //преобразуем его цветовой режим в оттенки серого
        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);

        //используем оператор Собеля, чтобы вычислить величину градиента серой картинки в вертикальном и горизонтальном направлениях.
        Imgproc.Sobel(grayImage, gradX, CvType.CV_32F, 1, 0);
        Imgproc.Sobel(gradX, gradY, CvType.CV_32F, 0, 1);

        //вычитаем y-градиент оператора Собеля из x-градиента.
        //После вычитания мы получаем изображение с высоким значением горизонтального градиента и низким значением вертикального
        Core.subtract(gradX, gradY, gradient);
        Core.convertScaleAbs(gradient, gradient);

        //устранить шум на изображении и сфокусироваться сугубо на области со штрихкодом.
        Imgproc.blur(gradient, blurImage, new Size(10.0, 9.0));
        Imgproc.threshold(blurImage, thresholdImage, 85.0, 85.0, Imgproc.THRESH_BINARY);

        //создадим прямоугольник с помощью getStructuringElement.
        // Ширина ядра больше его высоты, что позволяет нам перекрыть пространство между вертикальными полосками штрихкода.
        // произведем нашу морфологическую операцию, применив ядро к бинаризированному изображению,
        // замазывая пространство между полосками. И вы можно увидеть, что «пробелы» почти полностью закрыты
        Mat kernelImage = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(21.0, 7.0));
        Imgproc.morphologyEx(thresholdImage, closedImage, Imgproc.MORPH_CLOSE, kernelImage);

        //Тут мы делаем итерацию эрозии, за которой следует итерация дилатация.
        // Эрозия уберёт белые пиксели с изображения, удаляя мелкие блобы,
        // а дилатация не позволит крупным белым областям уменьшиться.
        // Удаленные во время размытия мелкие пятна во время растяжения не появятся вновь.
        Imgproc.erode(closedImage, closedImage, kernelImage, new Point(-1, -1), 9);
        Imgproc.dilate(closedImage, closedImage, kernelImage, new Point(-1, -1), 9);

//        Highgui.imwrite("F:\\barcodeResult.jpg", thresholdImage);

        //находим конторы на изображении
        List<MatOfPoint> matOfPoints = new ArrayList<>();
        Imgproc.findContours(closedImage, matOfPoints, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        //находим самый большой контур
        List<Point> points = new ArrayList<>();
        int maxSize = 0;
        for (int i = 0; i < matOfPoints.size(); i++) {
            List<Point> currentPoints = matOfPoints.get(i).toList();
            if (currentPoints.size() > maxSize) {
                maxSize = currentPoints.size();
                points = currentPoints;
            }

        }

        //находим самый маленький прямоугольник, чтобы в него вписывался контур
        MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
        matOfPoint2f.fromList(points);
        RotatedRect rect = Imgproc.minAreaRect(matOfPoint2f);

        //рисуем прямоугольник на первоначальной картике
        Point points2[] = new Point[4];
        rect.points(points2);
        for (int i = 0; i < 4; ++i) {
            Core.line(image, points2[i], points2[(i + 1) % 4], new Scalar(56, 0, 0), 5);
        }
//        Imgproc.drawContours(image, resultList, -1, new Scalar(0, 255, 0), 3);

        //записываем картинку в файл
        Highgui.imwrite("F:\\barcodeResult.jpg", image);
    }

}
