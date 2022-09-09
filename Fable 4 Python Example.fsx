#r"nuget: Fable.Core, 4.0.0-snake-island-alpha-007"

open Fable.Core

type IPhysicalDevice =
    abstract name: string
    abstract device_type: string

type IConfig =
    abstract list_physical_devices : string -> IPhysicalDevice array

type INDArray<'a> = 'a seq
type Image = INDArray<INDArray<uint8>>

type IMnist =
    abstract load_data: unit -> ((INDArray<Image> * INDArray<uint8>) * (INDArray<Image> * INDArray<uint8>))

type IDataSets =
    abstract mnist : IMnist

type IKeras =
    abstract datasets : IDataSets

type ITensorFlow =
    abstract config : IConfig
    abstract keras : IKeras

module NDArray =
    [<Emit("$2.reshape($0, $1)")>]
    let reshape<'a> : int -> int -> INDArray<'a> -> INDArray<'a> = nativeOnly

[<ImportAll("tensorflow")>]
let tensorflow: ITensorFlow = nativeOnly

tensorflow.config.list_physical_devices "CPU"

let ((image_train, label_train), (image_test, label_test)) = 
   tensorflow.keras.datasets.mnist.load_data()

printfn $"Image count: {(image_train |> Seq.length)}"
printfn "Line count: %A" (image_train |> Seq.head |> Seq.length)
printfn "Column count: %A" (image_train |> Seq.head |> Seq.head |> Seq.length)

// let image_train_flat = image_train |> NDArray.reshape<_> 60000 784 // / 255f

// let image_test_flat = (image_test?reshape(10000,784)) / 255f

// let inputs : obj = tensorflow?keras?Input(784)

// let dense : obj = tensorflow?keras?layers?Dense(10)

// let outputs : obj = dense?(inputs)

// let model : obj = tensorflow?keras?Model(inputs, outputs, "Digit_Recognition")

// model?summary()

// model?compile(
//     tensorflow?keras?optimizers?SGD(0.1f), 
//     tensorflow?keras?losses?SparseCategoricalCrossentropy(true),
//     "accuracy")

// model?fit((image_train_flat?numpy()),label_train,1)

// model?evaluate(image_test_flat?numpy(), label_test, 2)
