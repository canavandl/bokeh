import CDP from "chrome-remote-interface"
import chalk from "chalk"

import {format_exception} from "./format"
import type {Version} from "./types"
import {Exit, timeout, TimeoutError} from "./utils"

export type LogEntry = {
  level: "warning" | "error"
  text: string
}

export type Exception = {
  text: string
}

export type CDPClient = CDP.Client

export type CDPProtocol = {
  Emulation: CDP.EmulationProtocol
  Network: CDP.NetworkProtocol
  Browser: CDP.BrowserProtocol
  Page: CDP.PageProtocol
  DOM: CDP.DOMProtocol
  Runtime: CDP.RuntimeProtocol
  Log: CDP.LogProtocol
  Performance: CDP.PerformanceProtocol
}

export class Value<T> {
  constructor(public value: T) {}
}

export class Failure {
  constructor(public text: string) {}
}

export class Timeout {}

export const supported_chromium_version: Version = [141, 0, 7390, 54]

/**
 * Parses a version string into a Version tuple.
 * @param version - Version string like "141.0.7390.54"
 * @returns Version tuple [major, minor, patch, build] or null if invalid
 */
export function get_version_tuple(version: string): Version | null {
  const match = version.match(/(\d+)\.(\d+)\.(\d+)\.(\d+)/)
  if (match != null) {
    const [, a, b, c, d] = match.values()
    return [
      Number(a),
      Number(b),
      Number(c),
      Number(d),
    ]
  }
  return null
}

/**
 * Checks if the current Chromium version is supported.
 * Prints a warning if the version doesn't match the officially supported version.
 * @param current_chromium_version - Current version tuple or null if unknown
 */
export function check_version(current_chromium_version: Version | null): void {
  if (current_chromium_version == null) {
    const supported_str = supported_chromium_version.join(".")
    console.error(`${chalk.yellow("warning:")} unable to determine chromium version; officially supported version is ${supported_str}`)
    return
  }

  const [a, b, c, _d] = supported_chromium_version
  const [A, B, C, _D] = current_chromium_version

  if (a != A || b != B || c != C) {
    const supported_str = chalk.magenta(supported_chromium_version.join("."))
    const current_str = chalk.magenta(current_chromium_version.join("."))
    console.error(`${chalk.yellow("warning:")} ${current_str} is not supported; officially supported version is ${supported_str}`)
  }
}

export class BrowserManager {
  private client: CDPClient | null = null
  private protocol: CDPProtocol | null = null
  private entries: LogEntry[] = []
  private exceptions: Exception[] = []

  /**
   * Gets browser version information without establishing a full connection.
   * This is a static method that can be called before creating a BrowserManager instance.
   */
  static async get_version(port: number, host: string = "localhost"): Promise<{browser: string, protocol: string}> {
    const version = await CDP.Version({port, host})
    return {
      browser: version.Browser,
      protocol: version["Protocol-Version"],
    }
  }

  async connect(port: number, host: string = "localhost"): Promise<void> {
    this.client = await CDP({port, host})
    const {Emulation, Network, Browser, Page, DOM, Runtime, Log, Performance} = this.client

    this.protocol = {Emulation, Network, Browser, Page, DOM, Runtime, Log, Performance}

    // Set up exception and console handlers
    Runtime.exceptionThrown(({exceptionDetails}) => {
      this.exceptions.push(this.handle_exception(exceptionDetails))
    })

    Runtime.consoleAPICalled(({type, args}) => {
      if (type == "warning" || type == "error") {
        const text = args.map(({value}) => value ? value.toString() : "").join(" ")
        this.entries.push({level: type, text})
      }
    })

    Log.entryAdded(({entry}) => {
      const {level, text} = entry
      if (level == "warning" || level == "error") {
        this.entries.push({level, text})
      }
    })
  }

  async close(): Promise<void> {
    if (this.client != null) {
      await this.client.close()
      this.client = null
      this.protocol = null
    }
  }

  get_protocol(): CDPProtocol {
    if (this.protocol == null) {
      throw new Error("Browser not connected")
    }
    return this.protocol
  }

  get_entries(): LogEntry[] {
    return this.entries
  }

  get_exceptions(): Exception[] {
    return this.exceptions
  }

  clear_entries(): void {
    this.entries = []
  }

  clear_exceptions(): void {
    this.exceptions = []
  }

  private handle_exception(exceptionDetails: CDP.ExceptionDetails): Exception {
    const {text, stackTrace, exception, lineNumber, columnNumber} = exceptionDetails
    const formatted = format_exception({text, stackTrace, exception, lineNumber, columnNumber})
    return {text: formatted}
  }

  async initialize_page(url: string): Promise<void> {
    const {Emulation, Network, Browser, Page, DOM, Runtime, Log, Performance} = this.get_protocol()

    await Network.enable()
    await Network.setCacheDisabled({cacheDisabled: true})

    await Page.enable()
    await Page.navigate({url: "about:blank"})

    await DOM.enable({})

    await Runtime.enable()
    await Log.enable()
    await Performance.enable({timeDomain: "timeTicks"})

    await this.override_metrics()
    await Emulation.setFocusEmulationEnabled({enabled: true})

    await Browser.grantPermissions({
      permissions: ["clipboardReadWrite"],
    })

    const {errorText} = await Page.navigate({url})

    if (errorText != null) {
      this.fail(errorText)
    }

    if (this.exceptions.length != 0) {
      for (const exc of this.exceptions) {
        console.log(exc.text)
      }
      this.fail(`failed to load ${url}`)
    }

    await Page.loadEventFired()
  }

  async override_metrics(settings: {dpr?: number, scale?: number} = {}): Promise<void> {
    const {Emulation} = this.get_protocol()
    await Emulation.setDeviceMetricsOverride({
      width: 2000,
      height: 4000,
      deviceScaleFactor: settings.dpr ?? 1,
      mobile: false,
      scale: settings.scale ?? 1,
    })
  }

  async evaluate<T>(expression: string, eval_timeout: number = 10000): Promise<Value<T> | Failure | Timeout> {
    const {Runtime} = this.get_protocol()

    const output = await this.with_timeout(
      Runtime.evaluate({expression, returnByValue: true, awaitPromise: true}),
      eval_timeout,
    )

    if (output instanceof Timeout) {
      return output
    } else {
      const {result, exceptionDetails} = output
      if (exceptionDetails == null) {
        return new Value(result.value)
      } else {
        const {text} = this.handle_exception(exceptionDetails)
        return new Failure(text)
      }
    }
  }

  async is_ready(): Promise<boolean> {
    const expr = "typeof Bokeh !== 'undefined'"
    const result = await this.evaluate<boolean>(expr)
    return result instanceof Value && result.value
  }

  async capture_screenshot(bbox: {x: number, y: number, width: number, height: number}): Promise<Buffer> {
    const {Page} = this.get_protocol()
    const image = await Page.captureScreenshot({format: "png", clip: {...bbox, scale: 1}})
    return Buffer.from(image.data, "base64")
  }

  async discard_console_entries(): Promise<void> {
    const {Runtime} = this.get_protocol()
    await Runtime.discardConsoleEntries()
  }

  async get_metrics(): Promise<CDP.Metrics[]> {
    const {Performance} = this.get_protocol()
    const data = await Performance.getMetrics()
    return data.metrics
  }

  private async with_timeout<T>(promise: Promise<T>, wait: number): Promise<T | Timeout> {
    try {
      return await Promise.race([promise, timeout(wait)]) as T
    } catch (err) {
      if (err instanceof TimeoutError) {
        return new Timeout()
      } else {
        throw err
      }
    }
  }

  private fail(msg: string, code: number = 1): never {
    console.log(msg)
    throw new Exit(code)
  }
}
