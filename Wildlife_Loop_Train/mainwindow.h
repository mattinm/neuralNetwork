#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineEdit>
#include <QPalette>
#include <QCloseEvent>
#include <QProcess>
#include <QThread>
#include <QStringList>
#include <QStandardItemModel>
#include <QMap>

#include <string>

namespace Ui {
class MainWindow;
class Trainer;
class TrainerInfo;
}

class TrainerInfo{
public:
        std::string buildDir;
        QString netConfig, excludes = "--exclude=1000000 --exclude=999999 --exclude=21", msiLocations, trainMosaics, rootOutputLocation,
            testMosaics, outputLocation, trainIdxData, trainIdxLabel, trueBlobCounts, cnnName;
        unsigned int iterations, epochs;
};

class Trainer;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

signals:
    void closingWindow();
    void cancelSignal();

public slots:
    void disableInput();
    void enableInput();
    void cancelTrain();
    void closeEvent(QCloseEvent* event);

private slots:
    void run();
    void endRun();

    void updateTrainingLog(QString cnnName, QString filename);
    void updateBlobTable(QString filename);
    void updateCurrently(QString status);

    bool fileCheckEmptyAndExists(QLineEdit* textbox);
    bool folderCheckEmptyAndExists(QLineEdit* textbox);

    bool folderExists(QLineEdit *textbox);
    bool fileExists(QLineEdit *textbox);
    bool isNonEmpty(QLineEdit* textbox);

    void filePicker(QLineEdit* textbox, const char * filter = "*");
    void folderPicker(QLineEdit* textbox);

    void on_btnTrainingData_clicked();

    void on_btnTrainingLabel_clicked();

    void on_btnTrainMosaic_clicked();

    void on_btnTestMosaic_clicked();

    void on_btnMSILocations_clicked();

    void on_btnOutputLocation_clicked();

    void on_btnRun_clicked();

    void on_btnBuildDir_clicked();

    void on_btnNetConfig_clicked();

    void on_btnCancel_clicked();

    void on_btnTrueBlobCounts_clicked();


private:
    Ui::MainWindow *ui;
    QStandardItemModel *blobModel;
    QPalette errorPalette, standardPalette, missingPalette;
    bool currentlyRunning = false;
    Trainer *trainer;
    QThread *thr;
    TrainerInfo info;
    int curTrial, numTrials;
    QStringList csvFiles;
};

class Trainer : public QObject {
        Q_OBJECT

public:
    Trainer(MainWindow* parent, const TrainerInfo& info);
    ~Trainer();
    bool finishedWell() const;

public slots:
    void run();

signals:
    void endProcess();
    void finished();
    void error(QString error);
    void updateTrainingLog(QString cnnName, QString filename);
    void updateBlobTable(QString filename);
    void updateCurrently(QString status);


private:
    MainWindow *parent;
    TrainerInfo info;
    bool cancelTrain = false;
    bool finished_well = false;

    void trainCNN(const QString &outputCNN, const QString &oldCNN, const QString &curTrainIdxData, const QString &curTrainIdxLabel, const QString &termout);
    void runCNN(const QString& cnn, const QString& imageLocation, int stride, const QString& outloc);
    void blobCount(const QString& inputLocation, const QString& outputFilename);
    void compCNNObs(const QString& msi_locations, const QString& predImageDir, const QString& outputFilename, bool appendOutput = true);
    void compCNNObs(const QString& msi_locations, const QString& predImageDir, const QString& origImageDir, const QString& idx_base_name, const QString& outputFilename, bool appendOutput = true);
    void _compCNNObs(const QString& msi_locations, const QString& predImageDir, const QString& outputFilename, bool appendOutput, const QString& idxArgs);
    void combineIDXs(const QString& prev_data, const QString& prev_label, const QString& loop_data, const QString& loop_label, double percentFromOld, const QString& baseOutputName, const QString& terminalOutputFile, bool appendOutput = true);
    void compareBlobs(const QStringList &blob_counts, const QString& true_blob_counts, const QString &outputBaseName);
    QString getBestEpoch(QString output);

};

#endif // MAINWINDOW_H
