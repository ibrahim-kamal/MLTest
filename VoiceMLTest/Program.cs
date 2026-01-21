using Microsoft.ML;
using NWaves;
using NWaves.Audio;
using NWaves.FeatureExtractors;
using NWaves.FeatureExtractors.Options;
using NWaves.Operations;
using System.IO;
using System.Threading.Channels;

string path = Path.GetFullPath("features.csv");
var FolderPath = "C:/Users/ikamal/Downloads/voices/audio_speech_actors_01-24/";
List<List<float>> dataset = new List<List<float>>();
foreach (var FolderName in Directory.GetDirectories(FolderPath).Select(Path.GetFileName))
{
    foreach (var fileName in Directory.GetFiles($"{FolderPath}{FolderName}/"))
    {
        var emotion = float.Parse(fileName.Replace($"{FolderPath}{FolderName}","").Split('-')[2]);
        var result = ExtractMfcc(fileName);
        var features = result.ToList();
        features.Add(emotion);

        using (var writer = new StreamWriter(path, append: true))
        {
            writer.WriteLine(string.Join(",",features));
        }
        //dataset.Add(features);
    }
}


Console.ReadLine();
static float[] ExtractMfcc(string wavPath)
{
    using var fs = new FileStream(Path.GetFullPath(wavPath), FileMode.Open);
    // Load WAV file
    var waveFile = new WaveFile(fs);
    var signal = waveFile[Channels.Left];

    // Resample to 16kHz if needed
    if (signal.SamplingRate != 16000)
    {
        var resampler = new Resampler();
        signal = resampler.Resample(signal,16000);
    }
    
    // MFCC configuration
    var options = new MfccOptions
    {
        SamplingRate = 16000,
        FeatureCount = 13,
        FrameDuration = 0.025,   // 25ms
        HopDuration = 0.010      // 10ms
    };

    var extractor = new MfccExtractor(options);

    // Extract MFCC frames
    var mfccFrames = extractor.ComputeFrom(signal);

    // Convert to single vector (mean + std)
    var mean = Mean(mfccFrames);
    var std = Std(mfccFrames,mean);

    return mean.Concat(std).ToArray(); // 26 values
}

static float[] Mean(List<float[]> frames)
{
    int frameCount = frames.Count;
    int featureCount = frames[0].Length;

    var mean = new float[featureCount];

    foreach (var frame in frames)
    {
        for (int i = 0; i < featureCount; i++)
        {
            mean[i] += frame[i];
        }
    }

    for (int i = 0; i < featureCount; i++)
    {
        mean[i] /= frameCount;
    }

    return mean;
}

static float[] Std(List<float[]> frames, float[] mean)
{
    int frameCount = frames.Count;
    int featureCount = frames[0].Length;

    var std = new float[featureCount];

    foreach (var frame in frames)
    {
        for (int i = 0; i < featureCount; i++)
        {
            var diff = frame[i] - mean[i];
            std[i] += diff * diff;
        }
    }

    for (int i = 0; i < featureCount; i++)
    {
        std[i] = (float)Math.Sqrt(std[i] / frameCount);
    }

    return std;
}


public enum Emotion 
{
    neutral = 1,
    calm,
    happy,
    sad,
    angry,
    fearful,
    disgust,
    surprised,
}