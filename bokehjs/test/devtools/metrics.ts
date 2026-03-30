import type {BrowserManager} from "./browser.js"

export type MetricKeys = "JSEventListeners" | "Nodes" | "Resources" | "LayoutCount" | "RecalcStyleCount" | "JSHeapUsedSize" | "JSHeapTotalSize"

/**
 * Collects performance metrics during test execution.
 */
export class MetricsCollector {
  private metrics: {[key in MetricKeys]: number[]} = {
    JSEventListeners: [],
    Nodes: [],
    Resources: [],
    LayoutCount: [],
    RecalcStyleCount: [],
    JSHeapUsedSize: [],
    JSHeapTotalSize: [],
  }

  /**
   * Collects a single datapoint of metrics from the browser.
   *
   * @param browser - The browser manager to collect metrics from
   */
  async add_datapoint(browser: BrowserManager): Promise<void> {
    const data = await browser.get_metrics()
    for (const {name, value} of data) {
      switch (name) {
        case "JSEventListeners":
        case "Nodes":
        case "Resources":
        case "LayoutCount":
        case "RecalcStyleCount":
        case "JSHeapUsedSize":
        case "JSHeapTotalSize":
          this.metrics[name].push(value)
      }
    }
  }

  /**
   * Returns all collected metrics.
   *
   * @returns Object containing arrays of metric values for each metric type
   */
  get_metrics(): {[key in MetricKeys]: number[]} {
    return this.metrics
  }
}
