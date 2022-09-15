#r"nuget: Fable.Core, 4.0.0-theta-001"

open Fable.Core
open Fable.Core.PyInterop

type IPhysicalDevice =
    abstract name: string
    abstract device_type: string

type IConfig =
    abstract list_physical_devices : string -> IPhysicalDevice array

type NDArray =
    [<Emit("$0.reshape($1...)")>]
    member this.reshape([<ParamList>] args: int[]): NDArray = nativeOnly
    [<Emit("$0.ndim")>]
    member this.ndim : int = nativeOnly
    [<Emit("$0.shape")>]
    member this.shape : int array = nativeOnly
    [<Emit("$0 / $1")>]
    member this.``/``(arg : obj) : NDArray = nativeOnly
    [<Emit("$0")>]
    static member fromArray(arg : 'a[]): NDArray = nativeOnly
    [<Emit("$0")>]
    member this.toArray<'a>() : array<'a> = nativeOnly

type IMnist =
    abstract load_data: unit -> (NDArray * NDArray) * (NDArray * NDArray)

type IDataSets =
    abstract mnist : IMnist

type ITensors = ITensors

type IDense= IDense

type Layers =
    // it would be nice if this could be a named param
    [<NamedParams(fromIndex = 0)>]
    abstract Dense:
        units:int ->
            IDense

type IKeras =
    abstract datasets : IDataSets

    [<NamedParams(fromIndex = 0)>]
    abstract Input: 
        // it would be nice if this could be a param array
        shape:int[] ->
            ITensors

    abstract layers : Layers

type ITensorFlow =
    abstract config : IConfig
    abstract keras : IKeras

[<ImportAll("tensorflow")>]
let tensorflow: ITensorFlow = nativeOnly

tensorflow.config.list_physical_devices("CPU")

let ((image_train, label_train), (image_test, label_test)) = 
   tensorflow.keras.datasets.mnist.load_data()

// printfn $"Image train shape: {image_train.shape}"
// printfn $"Image test shape: {image_test.shape}"
// printfn $"Label train shape: {label_train.shape}"

let image_train_flat = (image_train.reshape [| 60000; 784 |]).``/`` 255

// printfn $"Image train flat shape: {image_train_flat.shape}"
// printfn $"First image: %A{image_train_flat.toArray<NDArray>() |> Array.head}" 

let image_test_flat = (image_test.reshape [| 10000; 784 |]).``/`` 255

let inputs = tensorflow.keras.Input(shape = [| 784 |])

let dense : obj = tensorflow.keras.layers.Dense(units = 10)

// let outputs : obj = dense?(inputs)

// let model : obj = tensorflow?keras?Model(inputs, outputs, "Digit_Recognition")

// model?summary()

// model?compile(
//     tensorflow?keras?optimizers?SGD(0.1f), 
//     tensorflow?keras?losses?SparseCategoricalCrossentropy(true),
//     "accuracy")

// model?fit((image_train_flat?numpy()),label_train,1)

// model?evaluate(image_test_flat?numpy(), label_test, 2)
